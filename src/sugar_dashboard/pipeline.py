from __future__ import annotations

import json
import math
import re
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


def _infer_month_from_report_file(report_file: str) -> str | None:
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
        r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[-_ ]+(20\d{2})",
        report_file,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    month = month_aliases[match.group(1).lower()]
    return f"{month} {match.group(2)}"


def _month_label_is_valid(month: str | None) -> bool:
    if not month:
        return False
    try:
        parsed_month = datetime.strptime(month, "%b %Y")
        if parsed_month.year < 2000 or parsed_month.year > 2100:
            return False
        return True
    except ValueError:
        return False


def _clean_cached_payload(payload: dict) -> dict:
    report_file = payload.get("report_file", "")
    extraction_payload = payload.get("extraction", {})
    if isinstance(extraction_payload, dict):
        inferred_month = _infer_month_from_report_file(report_file)
        if inferred_month and not _month_label_is_valid(extraction_payload.get("month")):
            extraction_payload["month"] = inferred_month

        report_date = extraction_payload.get("report_date")
        if isinstance(report_date, str):
            try:
                parsed_date = datetime.fromisoformat(report_date).date()
                if parsed_date.year < 2000 or parsed_date.year > 2100:
                    extraction_payload["report_date"] = None
            except ValueError:
                extraction_payload["report_date"] = None

    return payload


def _compute_direction(current: float | None, previous: float | None) -> str | None:
    if current is None or previous is None:
        return None
    if current > previous:
        return "Up"
    if current < previous:
        return "Down"
    return "Flat"


def _reports_are_adjacent_months(current: ProcessedReport | None, previous: ProcessedReport | None) -> bool:
    if current is None or previous is None:
        return False
    try:
        current_month = datetime.strptime(current.extraction.month, "%b %Y")
        previous_month = datetime.strptime(previous.extraction.month, "%b %Y")
    except ValueError:
        return False

    month_delta = (current_month.year - previous_month.year) * 12 + current_month.month - previous_month.month
    return month_delta == 1


def _derive_metrics(current: ProcessedReport | None, previous: ProcessedReport | None) -> DerivedMetrics:
    current_extraction = current.extraction if current else None
    previous_extraction = previous.extraction if _reports_are_adjacent_months(current, previous) else None

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
    payload = _clean_cached_payload(payload)
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
        inferred_month = _infer_month_from_report_file(report_path.name)
        if inferred_month and not _month_label_is_valid(extraction.month):
            extraction.month = inferred_month
        if extraction.report_date and (extraction.report_date.year < 2000 or extraction.report_date.year > 2100):
            extraction.report_date = None
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


def _replace_nan_with_none(value):
    if isinstance(value, dict):
        return {key: _replace_nan_with_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_nan_with_none(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def latest_row(frame: pd.DataFrame) -> DashboardRow | None:
    if frame.empty:
        return None
    return DashboardRow.model_validate(_replace_nan_with_none(frame.iloc[-1].to_dict()))
