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


MONTH_ALIASES = {
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


MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


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
    intent: Literal[
        "report_qa",
        "price_estimate",
        "dashboard_help",
        "general_domain_help",
        "out_of_scope",
        "data_lookup",
    ] = Field(
        description="Route the user question to the correct dashboard answer workflow."
    )
    target_month: str | None = Field(
        default=None,
        description="Canonical target month like 'Apr 2026' when the user asks for a month-specific answer.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class AnswerQualityCheck(BaseModel):
    passes: bool
    directly_answers_question: bool
    evidence_supports_answer: bool
    is_too_vague: bool
    missing_points: list[str] = Field(default_factory=list)
    critique: str


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
    months = _months_from_question(question)
    return months[0] if months else None


def _months_from_question(question: str) -> list[str]:
    month_mentions = list(re.finditer(rf"\b({MONTH_PATTERN})(?:\s+(20\d{{2}}))?\b", question, flags=re.IGNORECASE))
    explicit_years = [match.group(2) for match in month_mentions if match.group(2)]
    inferred_year = explicit_years[-1] if explicit_years else None
    months: list[str] = []
    for match in month_mentions:
        year = match.group(2) or inferred_year
        if not year:
            continue
        label = f"{MONTH_ALIASES[match.group(1).lower()]} {year}"
        if label not in months:
            months.append(label)
    return months


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
    target_months = _months_from_question(question)
    is_cross_month = len(target_months) > 1 or any(term in lowered for term in ("compare", "across", "from", "between", "evolve", "changed"))
    if target_months:
        if is_cross_month:
            scoped = _reports_for_comparison(target_months, sorted_reports)
            if scoped:
                return scoped
        return [report for report in sorted_reports if report.month == target_months[0]]

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
    terms = _tokens(question)
    lowered = question.lower()

    if _is_price_estimate_question(question):
        intent = "price_estimate"
    elif any(phrase in lowered for phrase in ("how does this dashboard", "how is this dashboard", "what does this dashboard", "how are the values", "how is pydantic", "what is pydantic")):
        intent = "dashboard_help"
    elif (
        "available reports" in lowered
        or "loaded reports" in lowered
        or "latest price" in lowered
        or "latest ny11" in lowered
        or ("compare" in terms and "india" in terms and bool(terms.intersection({"export", "exports", "outlook"})))
        or ("compare" in terms and bool(terms.intersection({"ny11", "price", "prices", "brent", "oil", "cane", "crush", "production", "mix"})))
    ):
        intent = "data_lookup"
    elif terms.intersection({"define", "definition", "mean", "means"}) or any(
        phrase in lowered
        for phrase in (
            "what is ny11",
            "what's ny11",
            "what is cane crush",
            "what is ethanol",
            "what does c/lb",
        )
    ):
        intent = "general_domain_help"
    elif not terms.intersection(DOMAIN_TERMS.union({"dashboard", "report", "reports", "price", "estimate", "expected"})):
        intent = "out_of_scope"
    else:
        intent = "report_qa"

    return QuestionIntent(
        intent=intent,
        target_month=_month_from_question(question),
        confidence=0.55,
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
                        "Use price_estimate only when the user asks for an expected, forecast, projected, "
                        "estimated, or forward-looking NY11/sugar price for a month after the latest loaded report. "
                        "Use data_lookup when the user asks for a specific dashboard value, latest loaded data, loaded reports, "
                        "or a direct numeric/text value available in the structured report fields. "
                        "Use data_lookup for comparisons when the metric is a structured dashboard value like NY11, Brent, cane crush, production, sugar mix, or India export outlook. "
                        "Use report_qa for qualitative comparisons such as broad risks, market tone, or why something changed. "
                        "Use dashboard_help when the user asks how the app, dashboard, extraction, RAG, evidence, or implementation works. "
                        "Use general_domain_help for stable commodity-market definitions or concept explanations that do not need report evidence. "
                        "Use out_of_scope for questions unrelated to sugar markets, commodity reports, or this dashboard. "
                        "Use report_qa for questions asking what loaded reports say, why something happened, "
                        "or any question that should be answered from document evidence. "
                        "If the user uses a relative target like next month, resolve it relative to the latest loaded report month. "
                        "Return the target_month in 'Mon YYYY' format when one is relevant."
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
    per_month_limit = max(1, min(top_k, 2))
    page_lookup = {
        (report.report_file, page.page_number): page.text
        for report in scoped_reports
        for page in report.page_text
    }

    ai_evidence = _retrieve_pageindex_with_ai(question, index_nodes, page_lookup, top_k=top_k)
    if ai_evidence:
        return _ensure_month_coverage(ai_evidence, index_nodes, page_lookup, per_month_limit=per_month_limit, top_k=top_k)

    page_records = [
        _page_node_to_record(node, page_lookup)
        for node in _flatten_page_index(index_nodes)
        if ".s" in node.node_id
    ]
    fallback = retrieve_evidence(question, page_records, top_k=top_k)
    fallback_evidence = [
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
    return _ensure_month_coverage(fallback_evidence, index_nodes, page_lookup, per_month_limit=per_month_limit, top_k=top_k)


def _ensure_month_coverage(
    evidence: list[RetrievedEvidence],
    index_nodes: list[PageIndexNode],
    page_lookup: dict[tuple[str, int], str],
    per_month_limit: int,
    top_k: int,
) -> list[RetrievedEvidence]:
    if len(index_nodes) <= 1:
        return evidence[:top_k]

    evidence_by_month: dict[str, list[RetrievedEvidence]] = {}
    for item in evidence:
        evidence_by_month.setdefault(item.record.month, []).append(item)

    covered: list[RetrievedEvidence] = []
    existing_source_ids = {item.record.source_id for item in evidence}
    for report_node in index_nodes:
        month_items = evidence_by_month.get(report_node.month, [])
        if month_items:
            covered.extend(month_items[:per_month_limit])
            continue

        section_nodes = [node for node in _flatten_page_index([report_node]) if ".s" in node.node_id]
        for node in section_nodes[:per_month_limit]:
            record = _page_node_to_record(node, page_lookup)
            if record.source_id in existing_source_ids:
                continue
            existing_source_ids.add(record.source_id)
            covered.append(
                RetrievedEvidence(
                    record=record,
                    retrieval_score=0.5,
                    rerank_score=0.5,
                    matched_terms=(),
                    search_path=f"{node.month} > {node.page_range_label} > {node.title}",
                    reasoning="Included to ensure each requested report month is represented in the cross-report comparison.",
                )
            )

    covered.extend(item for item in evidence if item.record.source_id not in {covered_item.record.source_id for covered_item in covered})
    return covered[:top_k]


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


def _simple_answer(
    question: str,
    answer: str,
    confidence: str,
    source_type: str,
    citation: str,
    can_answer: bool = True,
) -> RagAnswer:
    record = EvidenceRecord(
        source_id=f"{source_type.lower().replace(' ', '-')}:answer",
        source_type=source_type,
        title=source_type,
        month="N/A",
        page_number=None,
        text=answer,
        citation=citation,
    )
    evidence = [
        RetrievedEvidence(
            record=record,
            retrieval_score=1.0,
            rerank_score=1.0,
            matched_terms=tuple(sorted(_tokens(question).intersection(_tokens(answer)))),
            search_path=source_type,
            reasoning=confidence,
        )
    ]
    return RagAnswer(question=question, answer=answer, confidence=confidence, evidence=evidence, can_answer=can_answer)


def _answer_dashboard_help_question(question: str) -> RagAnswer:
    lowered = question.lower()
    if "rag" in lowered or "evidence" in lowered or "pageindex" in lowered:
        answer = (
            "The Ask a Question page uses an intent classifier, then a PageIndex-style report > page > section tree to retrieve relevant report sections before generating an answer. "
            "The Dashboard page itself does not use RAG; it renders cached structured extraction fields from the reports."
        )
    elif "pydantic" in lowered:
        answer = (
            "Pydantic defines the typed schemas for extracted report fields, cached processed reports, dashboard rows, and the question intent classifier. "
            "It validates LLM structured outputs and cached JSON before the dashboard uses them."
        )
    elif "values" in lowered or "dashboard" in lowered:
        answer = (
            "Dashboard values come from PyMuPDF text extraction, OpenAI structured extraction into Pydantic schemas, cached JSON files, and a Pandas dataframe rendered with Streamlit and Altair. "
            "KPI cards, supply drivers, and trade/risk panels read fields from that validated dataframe."
        )
    else:
        answer = (
            "The app has two main flows: the Dashboard renders cached structured report data, while Ask a Question uses intent routing plus section-level retrieval for report-grounded answers. "
            "Use Show raw evidence on the dashboard to inspect extracted JSON and snippets."
        )

    return _simple_answer(
        question=question,
        answer=answer,
        confidence="Answered from the dashboard implementation overview; no report retrieval was needed.",
        source_type="Dashboard help",
        citation="Application workflow",
    )


def _answer_general_domain_question(question: str) -> RagAnswer:
    lowered = question.lower()
    if "ny11" in lowered:
        answer = "NY11 usually refers to ICE raw sugar futures, a benchmark contract for world raw sugar prices quoted in cents per pound."
    elif "c/lb" in lowered or "cent" in lowered:
        answer = "c/lb means cents per pound, the common quotation unit for raw sugar futures such as NY11."
    elif "cane crush" in lowered or "crush" in lowered:
        answer = "Cane crush is the volume of sugarcane processed by mills; it is a key supply indicator because it drives sugar and ethanol output."
    elif "ethanol" in lowered:
        answer = "Ethanol matters for sugar because cane mills can shift some cane toward ethanol or sugar depending on relative prices, especially in Brazil."
    else:
        answer = (
            "I can explain stable sugar-market concepts such as NY11, cane crush, sugar mix, ethanol parity, imports, exports, and trade risks. "
            "For report-specific claims, ask a report-grounded question and I will retrieve the relevant section evidence."
        )

    return _simple_answer(
        question=question,
        answer=answer,
        confidence="Answered as general sugar-market background; no report retrieval was needed.",
        source_type="General domain help",
        citation="General commodity-market knowledge",
    )


def _answer_out_of_scope_question(question: str) -> RagAnswer:
    return _simple_answer(
        question=question,
        answer=(
            "I can only answer questions about the loaded sugar reports, the dashboard data, sugar-market concepts, or near-term NY11 estimates. "
            "Try asking about NY11, Brazil supply, India exports, Thailand production, oil/ethanol, trade risks, or how the dashboard works."
        ),
        confidence="No answer: the classifier routed this as outside the dashboard scope.",
        source_type="Out of scope",
        citation="Question intent router",
        can_answer=False,
    )


def _format_price(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f} c/lb"


def _reports_for_comparison(requested_months: list[str], reports: list[ProcessedReport]) -> list[ProcessedReport]:
    if len(requested_months) >= 2:
        start_month = _parse_month_label(requested_months[0])
        end_month = _parse_month_label(requested_months[-1])
        if start_month is None or end_month is None:
            return []
        if end_month < start_month:
            start_month, end_month = end_month, start_month
        return [
            report
            for report in reports
            if (parsed := _parse_month_label(report.month)) is not None and start_month <= parsed <= end_month
        ]
    if len(requested_months) == 1:
        return [report for report in reports if report.month == requested_months[0]]
    return reports


def _answer_comparison_lookup_question(question: str, selected_reports: list[ProcessedReport]) -> RagAnswer:
    terms = _tokens(question)
    if "india" in terms and bool(terms.intersection({"export", "exports", "outlook"})):
        points = []
        for report in selected_reports:
            note = report.extraction.india_exports_note or report.extraction.india_note or report.extraction.trade_summary
            points.append((report.month, report.report_file, note or "No India export note extracted."))

        answer = "India export outlook by month: " + " ".join(f"{month}: {note}" for month, _, note in points)
        record = EvidenceRecord(
            source_id="structured-data:india-export-comparison",
            source_type="Structured data lookup",
            title="India export outlook comparison",
            month=f"{points[0][0]} to {points[-1][0]}",
            page_number=None,
            text="; ".join(f"{month}: {note}" for month, _, note in points),
            citation="Cached structured extraction, India export outlook fields",
        )
        return RagAnswer(
            question=question,
            answer=answer,
            confidence="Answered from cached structured India export fields across the requested month range.",
            evidence=[
                RetrievedEvidence(
                    record=record,
                    retrieval_score=1.0,
                    rerank_score=1.0,
                    matched_terms=tuple(sorted(_tokens(question).intersection(_tokens(answer)))),
                    search_path=f"Structured data > India export outlook comparison > {points[0][0]} to {points[-1][0]}",
                    reasoning="The classifier routed this as a structured India export comparison, so matching month rows were compared.",
                )
            ],
            can_answer=True,
        )

    price_points = [
        (report.month, report.report_file, report.extraction.ny11_front_month_price)
        for report in selected_reports
        if report.extraction.ny11_front_month_price is not None
    ]
    if not price_points:
        return _simple_answer(
            question=question,
            answer="I found the requested reports, but they do not have extracted NY11 price values to compare.",
            confidence="No supported answer: cached NY11 values are missing for the requested comparison.",
            source_type="Structured data lookup",
            citation="Cached structured extraction",
            can_answer=False,
        )

    start_month, _, start_price = price_points[0]
    end_month, _, end_price = price_points[-1]
    change_abs = round(end_price - start_price, 2)
    change_pct = round((change_abs / start_price) * 100, 2) if start_price else None
    direction = "rose" if change_abs > 0 else "fell" if change_abs < 0 else "was unchanged"
    series = ", ".join(f"{month}: {price:.2f} c/lb" for month, _, price in price_points)
    pct_text = f" ({change_pct:+.2f}%)" if change_pct is not None else ""
    answer = (
        f"NY11 {direction} from {start_price:.2f} c/lb in {start_month} to {end_price:.2f} c/lb in {end_month}, "
        f"a change of {change_abs:+.2f} c/lb{pct_text}. Monthly extracted values: {series}."
    )

    record = EvidenceRecord(
        source_id="structured-data:ny11-comparison",
        source_type="Structured data lookup",
        title="NY11 comparison",
        month=f"{start_month} to {end_month}",
        page_number=None,
        text=series,
        citation="Cached structured extraction, NY11 front-month price",
    )
    return RagAnswer(
        question=question,
        answer=answer,
        confidence="Answered from cached structured dashboard fields across the requested month range.",
        evidence=[
            RetrievedEvidence(
                record=record,
                retrieval_score=1.0,
                rerank_score=1.0,
                matched_terms=tuple(sorted(_tokens(question).intersection(_tokens(answer)))),
                search_path=f"Structured data > NY11 comparison > {start_month} to {end_month}",
                reasoning="The classifier routed this as a structured data comparison, so all matching month rows were compared.",
            )
        ],
        can_answer=True,
    )


def _answer_data_lookup_question(question: str, reports: Iterable[ProcessedReport]) -> RagAnswer | None:
    report_list = sorted(reports, key=lambda item: item.extraction.month_sort_key)
    if not report_list:
        return None

    requested_months = _months_from_question(question)
    lowered = question.lower()
    is_comparison = "compare" in _tokens(question) or len(requested_months) > 1
    if is_comparison:
        selected_reports = _reports_for_comparison(requested_months, report_list)
        if selected_reports:
            return _answer_comparison_lookup_question(question, selected_reports)

    target_month = requested_months[0] if requested_months else None
    selected_reports = [report for report in report_list if report.month == target_month] if target_month else [report_list[-1]]
    if not selected_reports:
        return None
    report = selected_reports[-1]
    extraction = report.extraction

    if "available reports" in lowered or "loaded reports" in lowered:
        answer = "Loaded reports: " + ", ".join(f"{item.month} ({item.report_file})" for item in report_list) + "."
    elif "brent" in lowered or "oil" in lowered:
        answer = f"{report.month} Brent oil is {extraction.brent_oil if extraction.brent_oil is not None else 'N/A'} $/bbl in the cached structured extraction."
    elif "brazil" in lowered:
        answer = (
            f"For {report.month}, Brazil cane crush is {extraction.brazil_cane_crush_mmt or 'N/A'} mmt, "
            f"sugar production is {extraction.brazil_sugar_production_mmt or 'N/A'} mmt, and sugar mix is {extraction.brazil_sugar_mix_pct or 'N/A'}%."
        )
    elif "india" in lowered:
        answer = (
            f"For {report.month}, India current production is {extraction.india_current_production_mmt or 'N/A'} mmt, "
            f"final outlook is {extraction.india_final_outlook_mmt or 'N/A'} mmt, and exports note: {extraction.india_exports_note or 'N/A'}."
        )
    elif "thailand" in lowered:
        answer = (
            f"For {report.month}, Thailand production outlook is {extraction.thailand_production_outlook_mmt or 'N/A'} mmt "
            f"and ethanol diversion is {extraction.thailand_ethanol_diversion_kmt or 'N/A'} kmt."
        )
    elif "trade" in lowered or "risk" in lowered or "disruption" in lowered:
        answer = (
            f"For {report.month}, major trade disruption is {extraction.major_trade_disruption or 'None highlighted'}, "
            f"and trade summary is: {extraction.trade_summary or 'N/A'}."
        )
    elif "key driver" in lowered or "driver" in lowered:
        answer = f"For {report.month}, the extracted key driver is: {extraction.key_driver or 'N/A'}."
    elif "regime" in lowered:
        answer = f"For {report.month}, the extracted market regime is: {extraction.market_regime or 'N/A'}."
    else:
        answer = f"For {report.month}, NY11 front-month price is {_format_price(extraction.ny11_front_month_price)}."

    record = EvidenceRecord(
        source_id=f"{report.report_file}:structured-data-lookup",
        source_type="Structured data lookup",
        title="Cached structured extraction",
        month=report.month,
        page_number=None,
        text=answer,
        citation=f"{report.report_file}, cached structured extraction",
    )
    return RagAnswer(
        question=question,
        answer=answer,
        confidence="Answered from cached structured dashboard fields; no section retrieval was needed.",
        evidence=[
            RetrievedEvidence(
                record=record,
                retrieval_score=1.0,
                rerank_score=1.0,
                matched_terms=tuple(sorted(_tokens(question).intersection(_tokens(answer)))),
                search_path=f"Structured data > {report.month}",
                reasoning="The classifier routed this as a direct dashboard data lookup.",
            )
        ],
        can_answer=True,
    )


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


def _generate_ai_answer(
    question: str,
    evidence: list[RetrievedEvidence],
    corrective_feedback: str | None = None,
) -> str:
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
                        "Be direct and synthesize the finding. For compare/across/evolve questions, explicitly cover each requested month or period. "
                        "If the evidence is insufficient, say you cannot answer from the loaded reports."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Evidence:\n{_build_evidence_context(evidence)}\n\n"
                        f"{f'Corrective feedback to address: {corrective_feedback}\\n\\n' if corrective_feedback else ''}"
                        "Answer in 2-4 concise sentences. Mention the key evidence path or month-specific evidence when it matters."
                    ),
                },
            ],
        )
        return response.output_text.strip()
    except OpenAIError:
        return _make_answer(question, evidence)


def _heuristic_quality_check(question: str, answer: str, evidence: list[RetrievedEvidence]) -> AnswerQualityCheck:
    question_terms = _tokens(question)
    answer_terms = _tokens(answer)
    requested_months = _months_from_question(question)
    compare_question = bool(question_terms.intersection({"compare", "across", "between", "evolve", "changed"})) or len(requested_months) > 1
    missing_months = [month for month in requested_months if month.lower() not in answer.lower()] if compare_question else []
    vague_markers = ("appears", "may", "could", "uncertain", "potential", "conditions improve", "coming months")
    is_too_vague = compare_question and (len(answer) < 160 or any(marker in answer.lower() for marker in vague_markers))
    evidence_months = {item.record.month for item in evidence}
    missing_evidence_months = [
        month for month in requested_months if month not in evidence_months and not any(month in item.record.month for item in evidence)
    ] if compare_question else []
    directly_answers = bool(question_terms.intersection(answer_terms)) and not missing_months
    evidence_supports = bool(evidence) and not missing_evidence_months
    missing_points = missing_months + [f"Evidence for {month}" for month in missing_evidence_months]
    passes = directly_answers and evidence_supports and not is_too_vague
    return AnswerQualityCheck(
        passes=passes,
        directly_answers_question=directly_answers,
        evidence_supports_answer=evidence_supports,
        is_too_vague=is_too_vague,
        missing_points=missing_points,
        critique="Heuristic quality check used because the LLM critic was unavailable.",
    )


def _check_answer_quality(question: str, answer: str, evidence: list[RetrievedEvidence]) -> AnswerQualityCheck:
    settings = get_settings()
    if not settings.openai_api_key:
        return _heuristic_quality_check(question, answer, evidence)

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        response = client.responses.parse(
            model=settings.openai_model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a corrective RAG quality critic for a sugar-market dashboard. "
                        "Check whether the answer directly answers the question, is specific rather than vague, "
                        "and is supported by the supplied evidence paths/text. "
                        "For compare/across questions, require the answer and evidence to cover each requested month or period. "
                        "Do not grade style; grade correctness, specificity, and evidence support."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Draft answer:\n{answer}\n\n"
                        f"Evidence:\n{_build_evidence_context(evidence, max_items=8)}"
                    ),
                },
            ],
            text_format=AnswerQualityCheck,
        )
        if response.output_parsed is not None:
            return response.output_parsed
    except OpenAIError:
        pass

    return _heuristic_quality_check(question, answer, evidence)


def _make_quality_failure_answer(
    question: str,
    evidence: list[RetrievedEvidence],
    quality: AnswerQualityCheck,
) -> RagAnswer:
    missing = "; ".join(quality.missing_points) if quality.missing_points else quality.critique
    return RagAnswer(
        question=question,
        answer=(
            "I found evidence, but the generated answer did not pass the quality check. "
            f"Missing or weak points: {missing}. Try narrowing the question or asking for a specific month/driver."
        ),
        confidence=f"Corrective RAG blocked the answer: {quality.critique}",
        evidence=evidence,
        can_answer=False,
    )


def answer_report_question(question: str, reports: Iterable[ProcessedReport]) -> RagAnswer:
    report_list = list(reports)
    question_intent = classify_question_intent(question, report_list)
    if question_intent.intent == "price_estimate":
        estimate_answer = _answer_price_estimate_question(question, report_list, question_intent.target_month)
        if estimate_answer is not None:
            return estimate_answer
    if question_intent.intent == "dashboard_help":
        return _answer_dashboard_help_question(question)
    if question_intent.intent == "general_domain_help":
        return _answer_general_domain_question(question)
    if question_intent.intent == "out_of_scope":
        return _answer_out_of_scope_question(question)
    if question_intent.intent == "data_lookup":
        lookup_answer = _answer_data_lookup_question(question, report_list)
        if lookup_answer is not None:
            return lookup_answer

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

    answer = _generate_ai_answer(question, evidence)
    quality = _check_answer_quality(question, answer, evidence)
    if not quality.passes:
        corrective_feedback = (
            f"Critique: {quality.critique}. Missing points: {', '.join(quality.missing_points) or 'none listed'}. "
            "Revise to directly answer the question using only supplied evidence. "
            "If this is a comparison, cover each requested month/period explicitly."
        )
        answer = _generate_ai_answer(question, evidence, corrective_feedback=corrective_feedback)
        quality = _check_answer_quality(question, answer, evidence)
        if not quality.passes:
            return _make_quality_failure_answer(question, evidence, quality)

    return RagAnswer(
        question=question,
        answer=answer,
        confidence=(
            "AI-generated from the top retrieved report evidence and passed corrective quality review. "
            "Review the Evidence tab for source paths and citations."
        ),
        evidence=evidence,
        can_answer=True,
    )
