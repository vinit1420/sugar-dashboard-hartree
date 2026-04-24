from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

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


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    confidence: str
    evidence: list[RetrievedEvidence]
    can_answer: bool


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
    return re.sub(r"\s+", " ", value).strip()


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
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current)
    return chunks


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
    has_structured_support = evidence[0].record.source_type == "Structured extraction" and bool(matched_terms)
    has_specific_support = len(matched_terms) >= 2 or evidence[0].rerank_score >= 0.35 or has_structured_support
    return has_domain_match and has_specific_support


def _make_answer(question: str, evidence: list[RetrievedEvidence]) -> str:
    cited_points = []
    for item in evidence[:4]:
        text = item.record.text
        if len(text) > 420:
            text = text[:420].rsplit(" ", 1)[0] + "..."
        cited_points.append(f"- {text} ({item.record.citation})")

    return (
        "Based on the available ED&F Man sugar reports, the relevant evidence points to the following:\n\n"
        + "\n".join(cited_points)
    )


def answer_report_question(question: str, reports: Iterable[ProcessedReport]) -> RagAnswer:
    scoped_reports = _reports_for_question(question, reports)
    records = build_report_evidence(scoped_reports)
    evidence = retrieve_evidence(question, records)

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
        answer=_make_answer(question, evidence),
        confidence="Grounded in the top retrieved report snippets. Review the Evidence tab for source text and citations.",
        evidence=evidence,
        can_answer=True,
    )
