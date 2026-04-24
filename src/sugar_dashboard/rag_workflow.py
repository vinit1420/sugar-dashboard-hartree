from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    source_type: str
    title: str
    region: str
    period: str
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
    workflow_steps: list[str]


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "because",
    "did",
    "for",
    "how",
    "in",
    "is",
    "last",
    "month",
    "of",
    "the",
    "to",
    "was",
    "what",
    "why",
}


SAMPLE_EVIDENCE = [
    EvidenceRecord(
        source_id="erp-gulf-margin-mar-2026",
        source_type="ERP margin table",
        title="Gulf liquid feed margin bridge",
        region="Gulf",
        period="Mar 2026",
        text=(
            "Gulf liquid feed gross margin tightened by 185 basis points month over month. "
            "Average selling price was broadly flat, while delivered molasses input cost increased "
            "$18 per short ton and plant energy cost increased $4 per ton."
        ),
        citation="ERP margin bridge, Gulf, Mar 2026",
        weight=1.4,
    ),
    EvidenceRecord(
        source_id="procurement-molasses-gulf-mar-2026",
        source_type="Procurement note",
        title="Molasses procurement update",
        region="Gulf",
        period="Mar 2026",
        text=(
            "Gulf spot molasses availability remained tight as several suppliers held back offers. "
            "Delivered replacement values moved higher due to stronger export pull and limited nearby "
            "tank availability."
        ),
        citation="Procurement desk note, Molasses, Mar 2026",
        weight=1.2,
    ),
    EvidenceRecord(
        source_id="logistics-gulf-mar-2026",
        source_type="Logistics exception log",
        title="Gulf freight and terminal exceptions",
        region="Gulf",
        period="Mar 2026",
        text=(
            "Three Gulf inbound loads were delayed by terminal congestion and barge scheduling issues. "
            "Expedited trucking was used to protect customer service levels, adding an estimated $6 per "
            "ton to delivered cost on affected volume."
        ),
        citation="Logistics exception log, Gulf terminals, Mar 2026",
        weight=1.1,
    ),
    EvidenceRecord(
        source_id="market-sugar-ethanol-mar-2026",
        source_type="Market commentary",
        title="Sugar, ethanol, and molasses market context",
        region="Global",
        period="Mar 2026",
        text=(
            "Higher crude oil and firmer ethanol parity supported Brazilian mill incentives to favor "
            "ethanol at the margin. This reduced expectations for excess cane by-products and contributed "
            "to a tighter molasses tone in import-oriented regions."
        ),
        citation="Sugar and ethanol market commentary, Mar 2026",
        weight=1.0,
    ),
    EvidenceRecord(
        source_id="sales-gulf-mar-2026",
        source_type="Commercial note",
        title="Gulf customer pricing update",
        region="Gulf",
        period="Mar 2026",
        text=(
            "Commercial team held most customer liquid feed prices steady during March due to contract "
            "timing and competitive pressure. Price increases are under review for April renewals."
        ),
        citation="Commercial pricing note, Gulf, Mar 2026",
        weight=1.0,
    ),
    EvidenceRecord(
        source_id="erp-midwest-margin-mar-2026",
        source_type="ERP margin table",
        title="Midwest liquid feed margin bridge",
        region="Midwest",
        period="Mar 2026",
        text=(
            "Midwest liquid feed margin was flat month over month. Corn solubles cost declined enough "
            "to offset higher freight."
        ),
        citation="ERP margin bridge, Midwest, Mar 2026",
        weight=0.7,
    ),
]


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", value.lower()) if token not in STOP_WORDS}


def _record_text(record: EvidenceRecord) -> str:
    return " ".join([record.source_type, record.title, record.region, record.period, record.text])


def retrieve_evidence(
    question: str,
    records: Iterable[EvidenceRecord] = SAMPLE_EVIDENCE,
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    query_terms = _tokens(question)
    retrieved: list[RetrievedEvidence] = []

    for record in records:
        record_terms = _tokens(_record_text(record))
        matched_terms = tuple(sorted(query_terms.intersection(record_terms)))
        if not matched_terms:
            continue

        retrieval_score = len(matched_terms) / max(len(query_terms), 1)
        region_boost = 0.2 if "gulf" in query_terms and record.region.lower() == "gulf" else 0.0
        period_boost = 0.15 if "last" in question.lower() and record.period == "Mar 2026" else 0.0
        margin_boost = 0.15 if "margin" in record_terms else 0.0
        rerank_score = round((retrieval_score * record.weight) + region_boost + period_boost + margin_boost, 3)

        retrieved.append(
            RetrievedEvidence(
                record=record,
                retrieval_score=round(retrieval_score, 3),
                rerank_score=rerank_score,
                matched_terms=matched_terms,
            )
        )

    return sorted(retrieved, key=lambda item: item.rerank_score, reverse=True)[:top_k]


def generate_margin_answer(question: str) -> RagAnswer:
    evidence = retrieve_evidence(question)
    cited = {item.record.source_id: item.record.citation for item in evidence}

    answer = (
        "Gulf liquid feed margin tightened mainly because input and delivered costs rose faster than "
        "customer pricing. The ERP margin bridge shows a 185 bp month-over-month margin compression, "
        "with delivered molasses up $18/short ton and plant energy up $4/ton "
        f"({cited.get('erp-gulf-margin-mar-2026', 'ERP margin bridge')}). Procurement notes point to "
        "tight Gulf molasses availability, stronger export pull, and limited tank availability as the "
        "main input-cost drivers "
        f"({cited.get('procurement-molasses-gulf-mar-2026', 'Procurement note')}). Logistics records add "
        "that terminal congestion and barge scheduling forced expedited trucking on affected loads, "
        "adding roughly $6/ton to delivered cost "
        f"({cited.get('logistics-gulf-mar-2026', 'Logistics exception log')}). Market commentary provides "
        "the broader commodity link: higher crude oil and firmer ethanol parity supported Brazilian "
        "ethanol incentives, contributing to a tighter molasses tone "
        f"({cited.get('market-sugar-ethanol-mar-2026', 'Market commentary')}). Because commercial pricing "
        "was mostly held flat in March, the cost pressure was not fully passed through to customers "
        f"({cited.get('sales-gulf-mar-2026', 'Commercial pricing note')})."
    )

    workflow_steps = [
        "Filter evidence to the Gulf region and the latest monthly period.",
        "Retrieve ERP margin bridge, molasses procurement notes, logistics exceptions, and sugar/ethanol commentary.",
        "Rerank evidence toward margin movement, region match, current period, and source reliability.",
        "Generate a grounded explanation that cites each source and separates direct margin facts from market context.",
    ]

    return RagAnswer(
        question=question,
        answer=answer,
        confidence="High for direction of margin pressure; medium for exact attribution because some cost impacts affect only a subset of volume.",
        evidence=evidence,
        workflow_steps=workflow_steps,
    )
