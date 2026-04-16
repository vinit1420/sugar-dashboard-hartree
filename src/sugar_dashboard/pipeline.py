from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from sugar_dashboard.config import PROCESSED_DIR, REPORTS_DIR, get_settings
from sugar_dashboard.extractor import OpenAIReportExtractor
from sugar_dashboard.models import (
    DashboardRow,
    DerivedMetrics,
    ProcessedReport,
    build_dashboard_row,
)
from sugar_dashboard.pdf_ingestion import extract_pdf_pages


def _cache_path_for_report(report_path: Path) -> Path:
    return PROCESSED_DIR / f"{report_path.stem}.json"


def _compute_direction(current: float | None, previous: float | None) -> str | None:
    if current is None or previous is None:
        return None
    if current > previous:
        return "Up"
    if current < previous:
        return "Down"
    return "Flat"


def _derive_metrics(current: ProcessedReport | None, previous: ProcessedReport | None) -> DerivedMetrics:
    current_extraction = current.extraction if current else None
    previous_extraction = previous.extraction if previous else None

    current_price = current_extraction.ny11_front_month_price if current_extraction else None
    previous_price = previous_extraction.ny11_front_month_price if previous_extraction else None

    mom_abs: float | None = None
    mom_pct: float | None = None
    if current_price is not None and previous_price not in (None, 0):
        mom_abs = round(current_price - previous_price, 2)
        mom_pct = round(((current_price - previous_price) / previous_price) * 100, 2)

    regime_label = None
    if current_extraction:
        if current_extraction.market_regime:
            regime_label = current_extraction.market_regime
        elif current_extraction.key_driver:
            key_driver = current_extraction.key_driver.lower()
            if any(token in key_driver for token in ("conflict", "tight", "higher", "supportive", "bullish")):
                regime_label = "Supportive / upside risk"
            elif any(token in key_driver for token in ("comfortable", "oversupply", "bearish", "soft")):
                regime_label = "Soft / limited upside"
            else:
                regime_label = "Watchful / event-driven"
        elif current_extraction.macro_summary:
            regime_label = "Macro-driven"

    return DerivedMetrics(
        ny11_mom_change_abs=mom_abs,
        ny11_mom_change_pct=mom_pct,
        ny11_direction=_compute_direction(current_price, previous_price),
        brent_direction=_compute_direction(
            current_extraction.brent_oil if current_extraction else None,
            previous_extraction.brent_oil if previous_extraction else None,
        ),
        regime_label=regime_label,
    )


def _load_processed_report(cache_path: Path) -> ProcessedReport:
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload.pop("month", None)
    extraction_payload = payload.get("extraction", {})
    if isinstance(extraction_payload, dict):
        extraction_payload.pop("month_sort_key", None)
    return ProcessedReport.model_validate(payload)


def _save_processed_report(report: ProcessedReport, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.model_dump(mode="json")
    payload.pop("month", None)
    extraction_payload = payload.get("extraction", {})
    if isinstance(extraction_payload, dict):
        extraction_payload.pop("month_sort_key", None)
    cache_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_reports(force_reextract: bool = False) -> list[ProcessedReport]:
    settings = get_settings()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    report_paths = sorted(REPORTS_DIR.glob("*.pdf"))
    if not report_paths:
        return []

    extractor: OpenAIReportExtractor | None = None
    processed: list[ProcessedReport] = []

    for report_path in report_paths:
        cache_path = _cache_path_for_report(report_path)
        if cache_path.exists() and not force_reextract:
            processed.append(_load_processed_report(cache_path))
            continue

        if extractor is None:
            extractor = OpenAIReportExtractor(settings)

        ingested = extract_pdf_pages(report_path)
        extraction = extractor.extract(ingested)
        report = ProcessedReport(
            report_file=report_path.name,
            source_path=str(report_path),
            extraction=extraction,
            derived_metrics=DerivedMetrics(),
            extracted_at=datetime.utcnow(),
            page_count=len(ingested.pages),
            extracted_text_preview=ingested.combined_text[:4000],
            page_text=ingested.pages,
        )
        _save_processed_report(report, cache_path)
        processed.append(report)

    processed.sort(key=lambda item: item.extraction.month_sort_key)

    previous: ProcessedReport | None = None
    updated_reports: list[ProcessedReport] = []
    for report in processed:
        refreshed = report.model_copy(update={"derived_metrics": _derive_metrics(report, previous)})
        updated_reports.append(refreshed)
        _save_processed_report(refreshed, _cache_path_for_report(Path(report.source_path)))
        previous = refreshed

    return updated_reports


def reports_to_dataframe(reports: list[ProcessedReport]) -> pd.DataFrame:
    rows = [build_dashboard_row(report).model_dump(mode="json") for report in reports]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values("month_sort_key").reset_index(drop=True)
    return frame


def latest_row(frame: pd.DataFrame) -> DashboardRow | None:
    if frame.empty:
        return None
    return DashboardRow.model_validate(frame.iloc[-1].to_dict())
