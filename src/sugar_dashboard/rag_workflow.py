from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Literal

from openai import OpenAI
from openai import OpenAIError
from pydantic import BaseModel, Field

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


class QuestionIntent(BaseModel):
    intent: Literal["near_term_price_estimate", "report_qa"] = Field(
        description="Route the user question to a deterministic forecast estimate or normal report-grounded QA."
    )
    target_month: str | None = Field(
        default=None,
        description="Canonical target month like 'Apr 2026' when the user asks for a future estimate.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


@dataclass(frozen=True)
class PageIndexNode:
    title: str
    node_id: str
    start_page: int | None
    end_page: int | None
    summary: str
    report_file: str
    month: str
    text: str = ""
    children: tuple["PageIndexNode", ...] = ()

    @property
    def page_range_label(self) -> str:
        if self.start_page is None:
            return "summary"
        if self.end_page is None or self.end_page == self.start_page:
            return f"page {self.start_page}"
        return f"pages {self.start_page}-{self.end_page}"


@dataclass(frozen=True)
class LayoutBlock:
    text: str
    bbox: tuple[float, float, float, float]
    max_font_size: float
    is_bold: bool


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


def _month_scoring_terms(month: str) -> set[str]:
    aliases = {
        "Jan": {"jan", "january"},
        "Feb": {"feb", "february"},
        "Mar": {"mar", "march"},
        "Apr": {"apr", "april"},
        "May": {"may"},
        "Jun": {"jun", "june"},
        "Jul": {"jul", "july"},
        "Aug": {"aug", "august"},
        "Sep": {"sep", "sept", "september"},
        "Oct": {"oct", "october"},
        "Nov": {"nov", "november"},
        "Dec": {"dec", "december"},
    }
    parts = month.split()
    if len(parts) != 2:
        return _tokens(month)
    return aliases.get(parts[0], {parts[0].lower()}) | {parts[1]}


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


def _parse_month_label(month: str) -> datetime | None:
    try:
        return datetime.strptime(month, "%b %Y")
    except ValueError:
        return None


def _months_between(start: datetime, end: datetime) -> int:
    return (end.year - start.year) * 12 + end.month - start.month


def _latest_report_month(reports: Iterable[ProcessedReport]) -> str | None:
    sorted_reports = sorted(reports, key=lambda report: report.extraction.month_sort_key)
    return sorted_reports[-1].month if sorted_reports else None


def _classify_question_with_keywords(question: str) -> QuestionIntent:
    intent = "near_term_price_estimate" if _is_price_estimate_question(question) else "report_qa"
    return QuestionIntent(
        intent=intent,
        target_month=_month_from_question(question),
        confidence=0.55 if intent == "near_term_price_estimate" else 0.4,
        reasoning="Fallback keyword classifier used because the LLM classifier was unavailable.",
    )


def classify_question_intent(question: str, reports: Iterable[ProcessedReport]) -> QuestionIntent:
    report_list = list(reports)
    settings = get_settings()
    if not settings.openai_api_key:
        return _classify_question_with_keywords(question)

    loaded_months = ", ".join(report.month for report in sorted(report_list, key=lambda item: item.extraction.month_sort_key))
    latest_month = _latest_report_month(report_list) or "unknown"
    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.parse(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Classify commodity dashboard questions for routing. "
                        "Use near_term_price_estimate only when the user asks for an expected, forecast, projected, "
                        "estimated, or forward-looking NY11/sugar price for a month after the latest loaded report. "
                        "Use report_qa for questions asking what loaded reports say, why something happened, "
                        "or any question that should be answered from document evidence. "
                        "If the user uses a relative target like next month, resolve it relative to the latest loaded report month. "
                        "Return the target_month in 'Mon YYYY' format when intent is near_term_price_estimate."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Loaded report months: {loaded_months}\n"
                        f"Latest loaded report month: {latest_month}"
                    ),
                },
            ],
            text_format=QuestionIntent,
        )
        if response.output_parsed is not None:
            return response.output_parsed
    except OpenAIError:
        pass

    return _classify_question_with_keywords(question)


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


def _section_title(section_text: str, section_index: int) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", section_text.strip(), maxsplit=1)[0]
    first_sentence = re.sub(r"^\d+\s+(?:©\s+)?ED&F MAN\s+\d{4}\s+Monthly Sugar Note\s+", "", first_sentence)
    first_sentence = re.sub(
        r"^\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\s+",
        "",
        first_sentence,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^A-Za-z0-9 /&%$.-]+", " ", first_sentence)
    words = cleaned.split()
    if not words:
        return f"Section {section_index}"
    return " ".join(words[:10])


def _extract_layout_blocks(report: ProcessedReport, page_number: int) -> list[LayoutBlock]:
    try:
        import fitz
    except ImportError:
        return []

    try:
        with fitz.open(report.source_path) as document:
            page = document[page_number - 1]
            payload = page.get_text("dict")
    except Exception:
        return []

    blocks: list[LayoutBlock] = []
    for block in payload.get("blocks", []):
        if block.get("type") != 0:
            continue

        parts: list[str] = []
        max_font_size = 0.0
        is_bold = False
        for line in block.get("lines", []):
            line_parts = []
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                if not text:
                    continue
                line_parts.append(text)
                max_font_size = max(max_font_size, float(span.get("size", 0.0) or 0.0))
                font_name = str(span.get("font", "")).lower()
                is_bold = is_bold or "bold" in font_name
            if line_parts:
                parts.append(" ".join(line_parts))

        text = _clean_text(" ".join(parts))
        if not text or "disclaimer:" in text.lower() or "this report does not constitute" in text.lower():
            continue
        bbox = tuple(float(value) for value in block.get("bbox", (0, 0, 0, 0)))
        if len(bbox) != 4:
            bbox = (0.0, 0.0, 0.0, 0.0)
        blocks.append(LayoutBlock(text=text, bbox=bbox, max_font_size=max_font_size, is_bold=is_bold))

    return sorted(blocks, key=lambda item: (round(item.bbox[1] / 12) * 12, item.bbox[0]))


def _is_report_chrome(value: str) -> bool:
    lowered = value.lower()
    return (
        lowered.isdigit()
        or "monthly sugar note" in lowered and len(value) < 80
        or lowered.startswith("© ed&f man")
        or lowered.startswith("ed&f man")
    )


def _is_heading_block(block: LayoutBlock, median_font_size: float) -> bool:
    text = block.text.strip()
    lowered = text.lower()
    words = text.split()
    if re.match(r"^\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", lowered):
        return False
    known_heading_terms = (
        "markets",
        "focus",
        "brazil",
        "india",
        "thailand",
        "trade",
        "risk",
        "macro",
        "price",
        "ethanol",
        "oil",
        "freight",
        "specs",
        "funds",
        "imports",
        "exports",
    )
    if len(words) <= 8 and text[:1].isupper() and any(term in lowered for term in known_heading_terms):
        return True
    if len(text) <= 90 and block.max_font_size >= median_font_size + 1.2:
        return True
    return len(text) <= 70 and block.is_bold and len(words) <= 10


def _layout_sections(report: ProcessedReport, page) -> list[tuple[str, str]]:
    blocks = [block for block in _extract_layout_blocks(report, page.page_number) if not _is_report_chrome(block.text)]
    if not blocks:
        return []

    font_sizes = sorted(block.max_font_size for block in blocks if block.max_font_size > 0)
    median_font_size = font_sizes[len(font_sizes) // 2] if font_sizes else 0.0
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_parts: list[str] = []

    def flush_section() -> None:
        nonlocal current_title, current_parts
        section_text = _clean_text(" ".join(current_parts))
        if not _is_low_value_chunk(section_text):
            title = current_title or f"Section {len(sections) + 1}"
            sections.append((title, section_text))
        current_title = ""
        current_parts = []

    for block in blocks:
        if _is_heading_block(block, median_font_size):
            flush_section()
            current_title = _section_title(block.text, len(sections) + 1)
            current_parts = [block.text]
            continue
        current_parts.append(block.text)

    flush_section()
    return sections


def _text_sections(page) -> list[tuple[str, str]]:
    sections = _split_page_text(page.text, max_characters=900)
    if not sections:
        cleaned = _clean_text(page.text)
        sections = [cleaned] if cleaned else []
    return [(f"Section {index}", section) for index, section in enumerate(sections, start=1)]


def _build_section_nodes(report: ProcessedReport, report_index: int, page) -> tuple[PageIndexNode, ...]:
    sections = _layout_sections(report, page) or _text_sections(page)

    return tuple(
        PageIndexNode(
            title=title if not title.startswith("Section ") else f"Page {page.page_number} {title}",
            node_id=f"r{report_index:02d}.p{page.page_number:03d}.s{section_index:02d}",
            start_page=page.page_number,
            end_page=page.page_number,
            summary=section_text[:520],
            report_file=report.report_file,
            month=report.month,
            text=section_text,
        )
        for section_index, (title, section_text) in enumerate(sections, start=1)
        if not _is_low_value_chunk(section_text)
    )


def build_page_index(reports: Iterable[ProcessedReport]) -> list[PageIndexNode]:
    """Build a compact, PageIndex-style report > page > section hierarchy."""
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
                text=_clean_text(page.text),
                children=_build_section_nodes(report, report_index, page),
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
        for page_node in report_node.children:
            lines.append(
                f"  - {page_node.node_id} | {page_node.title} | {page_node.page_range_label} | "
                f"{page_node.summary[:max_summary_characters]}"
            )
            for section_node in page_node.children:
                lines.append(
                    f"    - {section_node.node_id} | {section_node.title} | "
                    f"{section_node.page_range_label} | {section_node.summary[:max_summary_characters]}"
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
    page_text = node.text
    if not page_text and node.start_page is not None:
        page_text = page_lookup.get((node.report_file, node.start_page), "")
    text = _clean_text(page_text) or node.summary
    source_type = "PageIndex section search" if ".s" in node.node_id else "PageIndex tree search"
    return EvidenceRecord(
        source_id=f"{node.report_file}:{node.node_id}",
        source_type=source_type,
        title=node.title,
        month=node.month,
        page_number=node.start_page,
        text=text,
        citation=f"{node.report_file}, {node.page_range_label}, {node.title}" if ".s" in node.node_id else f"{node.report_file}, {node.page_range_label}",
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
                        "Use the tree index to choose the most relevant report section nodes for the question. "
                        "Return only JSON with keys can_answer, reasoning, and nodes. "
                        "nodes must be an array of objects with node_id, relevance from 0 to 1, and reason. "
                        "Prefer precise section nodes ending in .sNN over page nodes; use page nodes only when no section is specific enough."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Tree index:\n{_format_page_index(index_nodes)}\n\n"
                        f"Select up to {top_k} relevant section nodes ending in .sNN."
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
        if node is None or not node.node_id.startswith("r") or ".s" not in node.node_id:
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
                search_path=f"{node.month} > {node.page_range_label} > {node.title}" if ".s" in node.node_id else f"{node.month} > {node.page_range_label}",
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
        if ".s" in node.node_id
    ]
    fallback = retrieve_evidence(question, page_records, top_k=top_k)
    return [
        RetrievedEvidence(
            record=item.record,
            retrieval_score=item.retrieval_score,
            rerank_score=item.rerank_score,
            matched_terms=item.matched_terms,
            search_path=f"{item.record.month} > page {item.record.page_number} > {item.record.title}",
            reasoning="Fallback lexical scoring over the PageIndex section summaries.",
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
    scoring_terms = set(query_terms)
    if target_month:
        scoring_terms -= _month_scoring_terms(target_month)
    if not scoring_terms:
        scoring_terms = set(query_terms)

    retrieved: list[RetrievedEvidence] = []
    for record in records:
        if target_month and record.month != target_month:
            continue

        record_terms = _tokens(" ".join([record.source_type, record.title, record.month, record.text]))
        matched_terms = tuple(sorted(scoring_terms.intersection(record_terms)))
        if not matched_terms:
            continue

        retrieval_score = len(matched_terms) / max(len(scoring_terms), 1)
        domain_boost = 0.12 if scoring_terms.intersection(DOMAIN_TERMS).intersection(record_terms) else 0.0
        month_boost = 0.08 if target_month and record.month == target_month else 0.0
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
    has_pageindex_support = evidence[0].record.source_type.startswith("PageIndex") and evidence[0].rerank_score >= 0.45
    has_structured_support = evidence[0].record.source_type == "Structured extraction" and bool(matched_terms)
    has_specific_support = len(matched_terms) >= 2 or evidence[0].rerank_score >= 0.35 or has_structured_support
    return has_pageindex_support or (has_domain_match and has_specific_support)


def _is_brazil_supply_question(question: str) -> bool:
    terms = _tokens(question)
    return "brazil" in terms and bool(terms.intersection({"supply", "production", "crop", "cane", "crush"}))


def _is_price_estimate_question(question: str) -> bool:
    terms = _tokens(question)
    has_price_term = bool(terms.intersection({"price", "prices", "ny11"}))
    has_estimate_term = bool(terms.intersection({"estimate", "expected", "forecast", "outlook", "projection"}))
    return has_price_term and has_estimate_term and _month_from_question(question) is not None


def _answer_price_estimate_question(
    question: str,
    reports: Iterable[ProcessedReport],
    target_month: str | None = None,
) -> RagAnswer | None:
    target_month = target_month or _month_from_question(question)
    target_date = _parse_month_label(target_month) if target_month else None
    if target_date is None:
        return None

    price_points = [
        (report, _parse_month_label(report.month), report.extraction.ny11_front_month_price)
        for report in sorted(reports, key=lambda item: item.extraction.month_sort_key)
    ]
    price_points = [
        (report, parsed_month, price)
        for report, parsed_month, price in price_points
        if parsed_month is not None and price is not None
    ]
    if len(price_points) < 2:
        return None

    latest_report, latest_month, latest_price = price_points[-1]
    if latest_month is None or latest_price is None:
        return None

    month_gap = _months_between(latest_month, target_date)
    if month_gap < 1 or month_gap > 3:
        return None

    prices = [price for _, _, price in price_points]
    deltas = [current - previous for previous, current in zip(prices, prices[1:])]
    recent_deltas = deltas[-3:] if len(deltas) >= 3 else deltas
    average_recent_delta = sum(recent_deltas) / len(recent_deltas)
    momentum_adjustment = average_recent_delta * month_gap
    point_estimate = round(latest_price + momentum_adjustment, 2)

    absolute_moves = [abs(delta) for delta in recent_deltas]
    average_move = sum(absolute_moves) / len(absolute_moves)
    half_range = max(0.45, average_move * 0.75)
    low_estimate = round(point_estimate - half_range, 2)
    high_estimate = round(point_estimate + half_range, 2)

    evidence_text = "; ".join(
        f"{report.month}: NY11 {price:.2f} c/lb"
        for report, _, price in price_points[-6:]
        if price is not None
    )
    evidence_record = EvidenceRecord(
        source_id=f"{latest_report.report_file}:ny11-trend-estimate",
        source_type="Trend estimate",
        title="Extracted NY11 price trend",
        month=latest_report.month,
        page_number=None,
        text=evidence_text,
        citation="Cached structured extraction, NY11 front-month price trend",
        weight=1.0,
    )
    evidence = [
        RetrievedEvidence(
            record=evidence_record,
            retrieval_score=1.0,
            rerank_score=1.0,
            matched_terms=tuple(sorted(_tokens(question).intersection({"price", "ny11", "estimate", "expected"}))),
            search_path=f"Structured prices > latest NY11 trend > {target_month} estimate",
            reasoning=(
                f"Used the latest extracted NY11 price ({latest_price:.2f} c/lb in {latest_report.month}) "
                f"and recent month-to-month momentum ({average_recent_delta:+.2f} c/lb/month)."
            ),
        )
    ]

    return RagAnswer(
        question=question,
        answer=(
            f"A simple trend-based estimate for NY11 in {target_month} is about {point_estimate:.2f} c/lb, "
            f"with a rough range of {low_estimate:.2f}-{high_estimate:.2f} c/lb. "
            f"This is an extrapolation from extracted report prices, not a reported April value or trading advice."
        ),
        confidence=(
            "Estimated from cached NY11 front-month prices because no loaded April 2026 report is available. "
            "Use as a directional dashboard estimate, not a market forecast."
        ),
        evidence=evidence,
        can_answer=True,
    )


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
    report_list = list(reports)
    question_intent = classify_question_intent(question, report_list)
    if question_intent.intent == "near_term_price_estimate":
        estimate_answer = _answer_price_estimate_question(question, report_list, question_intent.target_month)
        if estimate_answer is not None:
            return estimate_answer

    scoped_reports = _reports_for_question(question, report_list)
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
