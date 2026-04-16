from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


def _normalize_month_label(raw_value: str, year_hint: int | None = None) -> str:
    cleaned = raw_value.strip()
    for fmt in ("%b %Y", "%B %Y"):
        try:
            return datetime.strptime(cleaned, fmt).strftime("%b %Y")
        except ValueError:
            continue

    if year_hint is not None:
        for fmt in ("%b", "%B"):
            try:
                return datetime.strptime(cleaned, fmt).replace(year=year_hint).strftime("%b %Y")
            except ValueError:
                continue

    return cleaned


class SourceSnippets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ny11_front_month_price: str | None = None
    brazil_sugar_production_mmt: str | None = None
    india_final_outlook_mmt: str | None = None


class MarketReportExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    month: str
    report_date: date | None = None
    ny11_front_month_price: float | None = None
    ny11_price_change_pct: float | None = None
    london5_front_month_price: float | None = None
    brent_oil: float | None = None
    market_regime: str | None = None
    key_driver: str | None = None
    brazil_cane_crush_mmt: float | None = None
    brazil_sugar_production_mmt: float | None = None
    brazil_sugar_mix_pct: float | None = None
    brazil_note: str | None = None
    india_current_production_mmt: float | None = None
    india_final_outlook_mmt: float | None = None
    india_exports_note: str | None = None
    india_note: str | None = None
    thailand_production_outlook_mmt: float | None = None
    thailand_ethanol_diversion_kmt: float | None = None
    thailand_note: str | None = None
    major_trade_disruption: str | None = None
    market_positioning_note: str | None = None
    macro_summary: str | None = None
    supply_summary: str | None = None
    trade_summary: str | None = None
    executive_summary: str | None = None
    what_changed: list[str] | None = None
    why_it_matters: list[str] | None = None
    source_snippets: SourceSnippets = Field(default_factory=SourceSnippets)

    @model_validator(mode="after")
    def normalize_month(self) -> "MarketReportExtraction":
        year_hint = self.report_date.year if self.report_date else None
        self.month = _normalize_month_label(self.month, year_hint=year_hint)
        return self

    @computed_field
    @property
    def month_sort_key(self) -> str:
        try:
            parsed = datetime.strptime(self.month, "%b %Y")
            return parsed.strftime("%Y-%m")
        except ValueError:
            return self.month


class DerivedMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ny11_mom_change_abs: float | None = None
    ny11_mom_change_pct: float | None = None
    ny11_direction: str | None = None
    brent_direction: str | None = None
    regime_label: str | None = None


class PDFPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    text: str


class IngestedReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_name: str
    file_path: str
    pages: list[PDFPage]

    @property
    def combined_text(self) -> str:
        return "\n\n".join(
            f"[Page {page.page_number}]\n{page.text.strip()}" for page in self.pages if page.text.strip()
        )


class ProcessedReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_file: str
    source_path: str
    extraction: MarketReportExtraction
    derived_metrics: DerivedMetrics
    extracted_at: datetime
    page_count: int
    extracted_text_preview: str
    page_text: list[PDFPage]
    errors: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def month(self) -> str:
        return self.extraction.month


class DashboardRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    month: str
    month_sort_key: str
    report_date: date | None
    ny11_front_month_price: float | None
    london5_front_month_price: float | None
    brent_oil: float | None
    market_regime: str | None
    key_driver: str | None
    ny11_mom_change_abs: float | None
    ny11_mom_change_pct: float | None
    ny11_direction: str | None
    brent_direction: str | None
    regime_label: str | None
    brazil_cane_crush_mmt: float | None
    brazil_sugar_production_mmt: float | None
    brazil_sugar_mix_pct: float | None
    brazil_note: str | None
    india_current_production_mmt: float | None
    india_final_outlook_mmt: float | None
    india_exports_note: str | None
    india_note: str | None
    thailand_production_outlook_mmt: float | None
    thailand_ethanol_diversion_kmt: float | None
    thailand_note: str | None
    major_trade_disruption: str | None
    market_positioning_note: str | None
    macro_summary: str | None
    supply_summary: str | None
    trade_summary: str | None
    executive_summary: str | None
    what_changed: list[str] | None
    why_it_matters: list[str] | None
    source_snippets: dict[str, Any]
    extracted_text_preview: str
    report_file: str


def build_dashboard_row(report: ProcessedReport) -> DashboardRow:
    extraction = report.extraction
    derived = report.derived_metrics
    return DashboardRow(
        month=extraction.month,
        month_sort_key=extraction.month_sort_key,
        report_date=extraction.report_date,
        ny11_front_month_price=extraction.ny11_front_month_price,
        london5_front_month_price=extraction.london5_front_month_price,
        brent_oil=extraction.brent_oil,
        market_regime=extraction.market_regime,
        key_driver=extraction.key_driver,
        ny11_mom_change_abs=derived.ny11_mom_change_abs,
        ny11_mom_change_pct=derived.ny11_mom_change_pct,
        ny11_direction=derived.ny11_direction,
        brent_direction=derived.brent_direction,
        regime_label=derived.regime_label,
        brazil_cane_crush_mmt=extraction.brazil_cane_crush_mmt,
        brazil_sugar_production_mmt=extraction.brazil_sugar_production_mmt,
        brazil_sugar_mix_pct=extraction.brazil_sugar_mix_pct,
        brazil_note=extraction.brazil_note,
        india_current_production_mmt=extraction.india_current_production_mmt,
        india_final_outlook_mmt=extraction.india_final_outlook_mmt,
        india_exports_note=extraction.india_exports_note,
        india_note=extraction.india_note,
        thailand_production_outlook_mmt=extraction.thailand_production_outlook_mmt,
        thailand_ethanol_diversion_kmt=extraction.thailand_ethanol_diversion_kmt,
        thailand_note=extraction.thailand_note,
        major_trade_disruption=extraction.major_trade_disruption,
        market_positioning_note=extraction.market_positioning_note,
        macro_summary=extraction.macro_summary,
        supply_summary=extraction.supply_summary,
        trade_summary=extraction.trade_summary,
        executive_summary=extraction.executive_summary,
        what_changed=extraction.what_changed,
        why_it_matters=extraction.why_it_matters,
        source_snippets=extraction.source_snippets.model_dump(),
        extracted_text_preview=report.extracted_text_preview,
        report_file=report.report_file,
    )
