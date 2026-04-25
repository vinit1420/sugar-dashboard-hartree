from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI
from openai import OpenAIError

from sugar_dashboard.config import get_settings
from sugar_dashboard.models import ProcessedReport


SUGGESTED_QUESTIONS = [
    "Why did NY11 move higher in March 2026?",
    "What changed in Brazil supply across the latest reports?",
    "How did oil and ethanol influence the sugar market?",
    "What were the biggest trade or supply risks in the latest report?",
]


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    source_type: str
    title: str
    month: str
    page_number: int | None
    text: str
    citation: str
    weight: float = 1.0


@dataclass(frozen=True)
class RetrievedEvidence:
    record: EvidenceRecord
    retrieval_score: float
    rerank_score: float
    matched_terms: tuple[str, ...]
    search_path: str = ""
    reasoning: str = ""


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    confidence: str
    evidence: list[RetrievedEvidence]
    can_answer: bool


@dataclass(frozen=True)
class PageIndexNode:
    title: str
    node_id: str
    start_page: int | None
    end_page: int | None
    summary: str
    report_file: str
    month: str
    children: tuple["PageIndexNode", ...] = ()

    @property
    def page_range_label(self) -> str:
        if self.start_page is None:
            return "summary"
        if self.end_page is None or self.end_page == self.start_page:
            return f"page {self.start_page}"
        return f"pages {self.start_page}-{self.end_page}"


STOP_WORDS = {
    "a",
    "about",
    "across",
    "after",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "last",
    "latest",
    "market",
    "month",
    "of",
    "on",
    "or",
    "report",
    "reports",
    "s",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "why",
    "with",
}


DOMAIN_TERMS = {
    "sugar",
    "ny11",
    "london",
    "brent",
    "oil",
    "ethanol",
    "brazil",
    "india",
    "thailand",
    "crop",
    "cane",
    "crush",
    "production",
    "exports",
    "imports",
    "trade",
    "macro",
    "price",
    "prices",
    "regime",
    "driver",
    "drivers",
    "supply",
    "demand",
    "molasses",
    "weather",
}


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in STOP_WORDS}


def _month_from_question(question: str) -> str | None:
    month_aliases = {
        "jan": "Jan",
        "january": "Jan",
        "feb": "Feb",
        "february": "Feb",
        "mar": "Mar",
        "march": "Mar",
        "apr": "Apr",
        "april": "Apr",
        "may": "May",
        "jun": "Jun",
        "june": "Jun",
        "jul": "Jul",
        "july": "Jul",
        "aug": "Aug",
        "august": "Aug",
        "sep": "Sep",
        "sept": "Sep",
        "september": "Sep",
        "oct": "Oct",
        "october": "Oct",
        "nov": "Nov",
        "november": "Nov",
        "dec": "Dec",
        "december": "Dec",
    }
    match = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return f"{month_aliases[match.group(1).lower()]} {match.group(2)}"


def _reports_for_question(question: str, reports: Iterable[ProcessedReport]) -> list[ProcessedReport]:
    sorted_reports = sorted(reports, key=lambda report: report.extraction.month_sort_key)
    if not sorted_reports:
        return []

    lowered = question.lower()
    target_month = _month_from_question(question)
    if target_month:
        return [report for report in sorted_reports if report.month == target_month]

    if "latest reports" in lowered or "recent reports" in lowered:
        return sorted_reports[-3:]

    if "latest report" in lowered or "last month" in lowered or "most recent" in lowered:
        return sorted_reports[-1:]

    return sorted_reports


def _clean_text(value: str) -> str:
    cleaned = re.sub(
        r"Disclaimer: Any comments or opinions.*?accuracy\.",
        " ",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    cleaned = re.sub(r"This report does not constitute advice.*", " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"^\s*\d+\s+.*?Monthly Sugar Note\s+", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_low_value_chunk(value: str) -> bool:
    lowered = value.lower()
    return (
        len(value) < 80
        or "disclaimer:" in lowered
        or "this report does not constitute" in lowered
        or "should you seek to rely" in lowered
    )


def _split_page_text(text: str, max_characters: int = 850) -> list[str]:
    paragraphs = [_clean_text(part) for part in re.split(r"\n\s*\n", text) if _clean_text(part)]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_characters:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]

        for sentence in sentences:
            if not sentence:
                continue
            if current and len(current) + len(sentence) + 1 > max_characters:
                if not _is_low_value_chunk(current):
                    chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()

    if current and not _is_low_value_chunk(current):
        chunks.append(current)
    return [chunk for chunk in chunks if not _is_low_value_chunk(chunk)]


def _structured_summary_records(report: ProcessedReport) -> list[EvidenceRecord]:
    extraction = report.extraction
    summary_parts = [
        f"Market regime: {extraction.market_regime}" if extraction.market_regime else "",
        f"Key driver: {extraction.key_driver}" if extraction.key_driver else "",
        f"Macro summary: {extraction.macro_summary}" if extraction.macro_summary else "",
        f"Supply summary: {extraction.supply_summary}" if extraction.supply_summary else "",
        f"Trade summary: {extraction.trade_summary}" if extraction.trade_summary else "",
        f"Executive summary: {extraction.executive_summary}" if extraction.executive_summary else "",
    ]
    if extraction.what_changed:
        summary_parts.append("What changed: " + "; ".join(extraction.what_changed))
    if extraction.why_it_matters:
        summary_parts.append("Why it matters: " + "; ".join(extraction.why_it_matters))

    text = _clean_text(" ".join(part for part in summary_parts if part))
    if not text:
        return []

    return [
        EvidenceRecord(
            source_id=f"{report.report_file}:structured",
            source_type="Structured extraction",
            title="Extracted market summary",
            month=report.month,
            page_number=None,
            text=text,
            citation=f"{report.report_file}, extracted summary",
            weight=1.25,
        )
    ]


def build_report_evidence(reports: Iterable[ProcessedReport]) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for report in reports:
        records.extend(_structured_summary_records(report))
        for page in report.page_text:
            for chunk_index, chunk in enumerate(_split_page_text(page.text), start=1):
                records.append(
                    EvidenceRecord(
                        source_id=f"{report.report_file}:p{page.page_number}:c{chunk_index}",
                        source_type="Report text",
                        title=report.report_file,
                        month=report.month,
                        page_number=page.page_number,
                        text=chunk,
                        citation=f"{report.report_file}, page {page.page_number}",
                    )
                )
    return records


def _report_summary(report: ProcessedReport) -> str:
    summaries = _structured_summary_records(report)
    if summaries:
        return summaries[0].text
    return _clean_text(report.extracted_text_preview)[:1200]


def _page_summary(page_text: str, max_characters: int = 520) -> str:
    chunks = _split_page_text(page_text, max_characters=max_characters)
    if chunks:
        return chunks[0][:max_characters]
    return _clean_text(page_text)[:max_characters]


def build_page_index(reports: Iterable[ProcessedReport]) -> list[PageIndexNode]:
    """Build a compact, PageIndex-style hierarchy from cached report pages."""
    index: list[PageIndexNode] = []
    for report_index, report in enumerate(reports, start=1):
        page_nodes = tuple(
            PageIndexNode(
                title=f"{report.month} report page {page.page_number}",
                node_id=f"r{report_index:02d}.p{page.page_number:03d}",
                start_page=page.page_number,
                end_page=page.page_number,
                summary=_page_summary(page.text),
                report_file=report.report_file,
                month=report.month,
            )
            for page in report.page_text
            if _clean_text(page.text)
        )
        index.append(
            PageIndexNode(
                title=f"{report.month} - {report.report_file}",
                node_id=f"r{report_index:02d}",
                start_page=page_nodes[0].start_page if page_nodes else None,
                end_page=page_nodes[-1].end_page if page_nodes else None,
                summary=_report_summary(report),
                report_file=report.report_file,
                month=report.month,
                children=page_nodes,
            )
        )
    return index


def _flatten_page_index(nodes: Iterable[PageIndexNode]) -> list[PageIndexNode]:
    flattened: list[PageIndexNode] = []
    for node in nodes:
        flattened.append(node)
        flattened.extend(_flatten_page_index(node.children))
    return flattened


def _format_page_index(nodes: Iterable[PageIndexNode], max_summary_characters: int = 420) -> str:
    lines: list[str] = []
    for report_node in nodes:
        lines.append(
            f"- {report_node.node_id} | {report_node.month} | {report_node.title} | "
            f"{report_node.page_range_label} | {report_node.summary[:max_summary_characters]}"
        )
        for child in report_node.children:
            lines.append(
                f"  - {child.node_id} | {child.title} | {child.page_range_label} | "
                f"{child.summary[:max_summary_characters]}"
            )
    return "\n".join(lines)


def _parse_json_object(value: str) -> dict:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _page_node_to_record(node: PageIndexNode, page_lookup: dict[tuple[str, int], str]) -> EvidenceRecord:
    page_text = ""
    if node.start_page is not None:
        page_text = page_lookup.get((node.report_file, node.start_page), "")
    text = _clean_text(page_text) or node.summary
    return EvidenceRecord(
        source_id=f"{node.report_file}:{node.node_id}",
        source_type="PageIndex tree search",
        title=node.title,
        month=node.month,
        page_number=node.start_page,
        text=text,
        citation=f"{node.report_file}, {node.page_range_label}",
        weight=1.35,
    )


def _retrieve_pageindex_with_ai(
    question: str,
    index_nodes: list[PageIndexNode],
    page_lookup: dict[tuple[str, int], str],
    top_k: int,
) -> list[RetrievedEvidence]:
    settings = get_settings()
    if not settings.openai_api_key:
        return []

    node_by_id = {node.node_id: node for node in _flatten_page_index(index_nodes)}
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You perform PageIndex-style retrieval over ED&F Man sugar reports. "
                        "Use the tree index to choose the most relevant page nodes for the question. "
                        "Return only JSON with keys can_answer, reasoning, and nodes. "
                        "nodes must be an array of objects with node_id, relevance from 0 to 1, and reason. "
                        "Prefer precise page nodes over report root nodes."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Tree index:\n{_format_page_index(index_nodes)}\n\n"
                        f"Select up to {top_k} relevant page nodes."
                    ),
                },
            ],
        )
    except OpenAIError:
        return []

    payload = _parse_json_object(response.output_text)
    selected_nodes = payload.get("nodes", [])
    if not isinstance(selected_nodes, list):
        return []

    retrieved: list[RetrievedEvidence] = []
    for selected in selected_nodes:
        if not isinstance(selected, dict):
            continue
        node_id = str(selected.get("node_id", "")).strip()
        node = node_by_id.get(node_id)
        if node is None or not node.node_id.startswith("r") or ".p" not in node.node_id:
            continue
        try:
            relevance = float(selected.get("relevance", 0.0))
        except (TypeError, ValueError):
            relevance = 0.0
        reason = str(selected.get("reason", "")).strip()
        retrieved.append(
            RetrievedEvidence(
                record=_page_node_to_record(node, page_lookup),
                retrieval_score=round(max(0.0, min(relevance, 1.0)), 3),
                rerank_score=round(max(0.0, min(relevance, 1.0)) + 0.2, 3),
                matched_terms=tuple(sorted(_tokens(question).intersection(_tokens(f"{node.title} {node.summary}")))),
                search_path=f"{node.month} > {node.page_range_label}",
                reasoning=reason or str(payload.get("reasoning", "")).strip(),
            )
        )

    return sorted(retrieved, key=lambda item: item.rerank_score, reverse=True)[:top_k]


def retrieve_pageindex_evidence(
    question: str,
    reports: Iterable[ProcessedReport],
    top_k: int = 6,
) -> list[RetrievedEvidence]:
    scoped_reports = list(reports)
    index_nodes = build_page_index(scoped_reports)
    page_lookup = {
        (report.report_file, page.page_number): page.text
        for report in scoped_reports
        for page in report.page_text
    }

    ai_evidence = _retrieve_pageindex_with_ai(question, index_nodes, page_lookup, top_k=top_k)
    if ai_evidence:
        return ai_evidence

    page_records = [
        _page_node_to_record(node, page_lookup)
        for node in _flatten_page_index(index_nodes)
        if ".p" in node.node_id
    ]
    fallback = retrieve_evidence(question, page_records, top_k=top_k)
    return [
        RetrievedEvidence(
            record=item.record,
            retrieval_score=item.retrieval_score,
            rerank_score=item.rerank_score,
            matched_terms=item.matched_terms,
            search_path=f"{item.record.month} > page {item.record.page_number}",
            reasoning="Fallback lexical scoring over the PageIndex tree summaries.",
        )
        for item in fallback
    ]


def retrieve_evidence(
    question: str,
    records: Iterable[EvidenceRecord],
    top_k: int = 6,
) -> list[RetrievedEvidence]:
    query_terms = _tokens(question)
    if not query_terms:
        return []

    target_month = _month_from_question(question)
    retrieved: list[RetrievedEvidence] = []
    for record in records:
        if target_month and record.month != target_month:
            continue

        record_terms = _tokens(" ".join([record.source_type, record.title, record.month, record.text]))
        matched_terms = tuple(sorted(query_terms.intersection(record_terms)))
        if not matched_terms:
            continue

        retrieval_score = len(matched_terms) / max(len(query_terms), 1)
        domain_boost = 0.12 if query_terms.intersection(DOMAIN_TERMS).intersection(record_terms) else 0.0
        month_boost = 0.08 if any(term in record.month.lower() for term in query_terms) else 0.0
        structured_boost = 0.08 if record.source_type == "Structured extraction" else 0.0
        rerank_score = round((retrieval_score * record.weight) + domain_boost + month_boost + structured_boost, 3)

        retrieved.append(
            RetrievedEvidence(
                record=record,
                retrieval_score=round(retrieval_score, 3),
                rerank_score=rerank_score,
                matched_terms=matched_terms,
            )
        )

    return sorted(retrieved, key=lambda item: item.rerank_score, reverse=True)[:top_k]


def _has_enough_support(question: str, evidence: list[RetrievedEvidence]) -> bool:
    if not evidence:
        return False

    query_terms = _tokens(question)
    matched_terms = set(evidence[0].matched_terms)
    has_domain_match = bool(query_terms.intersection(DOMAIN_TERMS))
    has_pageindex_support = evidence[0].record.source_type == "PageIndex tree search" and evidence[0].rerank_score >= 0.45
    has_structured_support = evidence[0].record.source_type == "Structured extraction" and bool(matched_terms)
    has_specific_support = len(matched_terms) >= 2 or evidence[0].rerank_score >= 0.35 or has_structured_support
    return has_pageindex_support or (has_domain_match and has_specific_support)


def _is_brazil_supply_question(question: str) -> bool:
    terms = _tokens(question)
    return "brazil" in terms and bool(terms.intersection({"supply", "production", "crop", "cane", "crush"}))


def _find_brazil_supply_evidence(reports: Iterable[ProcessedReport]) -> list[RetrievedEvidence]:
    records = build_report_evidence(reports)
    wanted_phrases = (
        "brazil c/s",
        "focus - brazil",
        "focus – brazil",
        "ethanol",
        "sugar production",
        "sugar mix",
        "atr",
    )
    evidence: list[RetrievedEvidence] = []

    for record in records:
        haystack = f"{record.title} {record.text}".lower()
        if "brazil" not in haystack:
            continue
        matched_terms = tuple(
            sorted(
                term
                for term in ("brazil", "cane", "crush", "ethanol", "production", "mix", "atr", "stocks")
                if term in haystack
            )
        )
        if not matched_terms:
            continue

        phrase_score = sum(0.18 for phrase in wanted_phrases if phrase in haystack)
        page_score = 0.1 if record.page_number in (2, 3, 5, None) else 0.0
        month_score = {"Jan 2026": 0.05, "Feb 2026": 0.12, "Mar 2026": 0.2}.get(record.month, 0.0)
        rerank_score = round(0.2 + phrase_score + page_score + month_score + (0.05 * len(matched_terms)), 3)
        evidence.append(
            RetrievedEvidence(
                record=record,
                retrieval_score=round(min(len(matched_terms) / 8, 1), 3),
                rerank_score=rerank_score,
                matched_terms=matched_terms,
            )
        )

    return sorted(evidence, key=lambda item: item.rerank_score, reverse=True)[:6]


def _answer_brazil_supply_question(question: str, reports: Iterable[ProcessedReport]) -> RagAnswer:
    scoped_reports = _reports_for_question(question, reports)
    evidence = _find_brazil_supply_evidence(scoped_reports)
    if not evidence:
        return RagAnswer(
            question=question,
            answer="I can't answer that from the ED&F Man sugar reports currently loaded in the dashboard.",
            confidence="No supported answer: Brazil supply evidence was not found in the loaded reports.",
            evidence=[],
            can_answer=False,
        )

    return RagAnswer(
        question=question,
        answer=_generate_ai_answer(question, evidence),
        confidence="AI-generated from Brazil C/S evidence in the latest reports. Review the Evidence tab for source text and citations.",
        evidence=evidence,
        can_answer=True,
    )


def _make_answer(question: str, evidence: list[RetrievedEvidence]) -> str:
    cited_points = []
    for item in evidence[:3]:
        text = item.record.text
        if "Monthly Sugar Note " in text[:80]:
            text = text.split("Monthly Sugar Note ", 1)[1]
        if len(text) > 260:
            text = text[:260].rsplit(" ", 1)[0] + "..."
        cited_points.append(f"- {text} ({item.record.citation})")

    return (
        "Short answer: the reports point to the following supported takeaways:\n\n"
        + "\n".join(cited_points)
    )


def _build_evidence_context(evidence: list[RetrievedEvidence], max_items: int = 5) -> str:
    context_parts = []
    for index, item in enumerate(evidence[:max_items], start=1):
        text = item.record.text
        if "Monthly Sugar Note " in text[:80]:
            text = text.split("Monthly Sugar Note ", 1)[1]
        context_parts.append(
            "\n".join(
                [
                    f"Evidence {index}",
                    f"Citation: {item.record.citation}",
                    f"Search path: {item.search_path}" if item.search_path else "",
                    f"Retrieval reasoning: {item.reasoning}" if item.reasoning else "",
                    f"Text: {text[:1200]}",
                ]
            )
        )
    return "\n\n".join(context_parts)


def _generate_ai_answer(question: str, evidence: list[RetrievedEvidence]) -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        return _make_answer(question, evidence)

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.create(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a commodity market analyst. Answer using only the supplied ED&F Man report evidence. "
                        "Write 2 concise sentences. Be direct and synthesize the finding; do not list evidence bullets. "
                        "If the evidence is insufficient, say you cannot answer from the loaded reports."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Evidence:\n{_build_evidence_context(evidence)}\n\n"
                        "Answer in 2 concise sentences."
                    ),
                },
            ],
        )
        return response.output_text.strip()
    except OpenAIError:
        return _make_answer(question, evidence)


def answer_report_question(question: str, reports: Iterable[ProcessedReport]) -> RagAnswer:
    scoped_reports = _reports_for_question(question, reports)
    if _is_brazil_supply_question(question):
        return _answer_brazil_supply_question(question, scoped_reports)

    evidence = retrieve_pageindex_evidence(question, scoped_reports)

    if not _has_enough_support(question, evidence):
        return RagAnswer(
            question=question,
            answer=(
                "I can't answer that from the ED&F Man sugar reports currently loaded in the dashboard. "
                "Try asking about sugar prices, NY11, Brazil, India, Thailand, oil/ethanol, trade flows, macro drivers, or supply risks."
            ),
            confidence="No supported answer: the retrieved report evidence was not relevant enough.",
            evidence=evidence,
            can_answer=False,
        )

    return RagAnswer(
        question=question,
        answer=_generate_ai_answer(question, evidence),
        confidence="AI-generated from the top retrieved report evidence. Review the Evidence tab for source text and citations.",
        evidence=evidence,
        can_answer=True,
    )
