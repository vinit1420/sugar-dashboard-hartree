from __future__ import annotations

import json
import re
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from sugar_dashboard.pipeline import latest_row, load_reports, reports_to_dataframe
from sugar_dashboard.rag_workflow import SUGGESTED_QUESTIONS, answer_report_question


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        .hero {
            background: linear-gradient(135deg, #0f2747 0%, #1f6feb 100%);
            border-radius: 20px;
            color: white;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 18px 40px rgba(15, 39, 71, 0.16);
        }
        .hero h1 {
            margin: 0;
            font-size: 2rem;
        }
        .hero p {
            margin: 0.45rem 0 0;
            color: rgba(255,255,255,0.85);
        }
        .metric-card {
            background: white;
            border: 1px solid rgba(16, 32, 51, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 26px rgba(16, 32, 51, 0.06);
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #5c6b80;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
            color: #102033;
        }
        .metric-help {
            margin-top: 0.35rem;
            color: #607089;
            font-size: 0.92rem;
        }
        .section-card {
            background: white;
            border-radius: 18px;
            border: 1px solid rgba(16, 32, 51, 0.08);
            padding: 1.1rem 1.15rem;
            box-shadow: 0 10px 26px rgba(16, 32, 51, 0.06);
            height: 100%;
        }
        .eyebrow {
            color: #607089;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.74rem;
            margin-bottom: 0.45rem;
        }
        .bullet-list {
            margin: 0;
            padding-left: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <div class="eyebrow">{title}</div>
            <div>{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_number(value: float | int | None, suffix: str = "", decimals: int = 1) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{suffix}"


def _format_change(value: float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and pd.isna(value):
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


def _optional_text(value: Any) -> str | None:
    if _is_missing(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _text_value(value: Any, fallback: str) -> str:
    return _optional_text(value) or fallback


def _list_value(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if not _is_missing(item)]


def _build_trend_chart(frame: pd.DataFrame) -> alt.Chart:
    plot_frame = frame.copy()
    plot_frame["report_month"] = pd.Categorical(
        plot_frame["month"],
        categories=plot_frame["month"].tolist(),
        ordered=True,
    )

    base = alt.Chart(plot_frame).encode(
        x=alt.X("report_month:N", title="Month", sort=plot_frame["month"].tolist()),
    )

    ny11_line = base.mark_line(point=True, strokeWidth=3, color="#1f6feb").encode(
        y=alt.Y("ny11_front_month_price:Q", title="NY11 front-month price (c/lb)"),
        tooltip=[
            alt.Tooltip("month:N", title="Month"),
            alt.Tooltip("ny11_front_month_price:Q", title="NY11 (c/lb)", format=".2f"),
            alt.Tooltip("brent_oil:Q", title="Brent ($/bbl)", format=".1f"),
        ],
    )

    london_line = base.mark_line(point=True, strokeDash=[6, 4], strokeWidth=2, color="#f59e0b").encode(
        y=alt.Y("brent_oil:Q", title="Brent oil ($/bbl)"),
    )

    return alt.layer(ny11_line, london_line).resolve_scale(y="independent").properties(height=320)


def _build_market_regime_display(selected: pd.Series) -> tuple[str, str]:
    extracted_regime = _optional_text(selected["market_regime"])
    derived_regime = _optional_text(selected["regime_label"])
    context = " ".join(
        part
        for part in [
            _optional_text(selected["macro_summary"]),
            _optional_text(selected["key_driver"]),
            _optional_text(selected["trade_summary"]),
        ]
        if part
    ).lower()

    bullish_signals = (
        "price higher",
        "prices higher",
        "pushing prices higher",
        "upside risk",
        "tight",
        "supportive",
        "bullish",
        "move higher",
        "rally",
        "raising expectations",
    )
    bearish_signals = (
        "oversupply",
        "comfortable",
        "soft",
        "bearish",
        "limited upside",
        "downside",
        "weaker prices",
    )

    extracted_regime_lc = (extracted_regime or "").strip().lower()
    bullish_context = any(signal in context for signal in bullish_signals)
    bearish_context = any(signal in context for signal in bearish_signals)

    if extracted_regime_lc == "bearish" and bullish_context:
        primary = "Supportive / upside risk"
    elif extracted_regime_lc == "bullish" and bearish_context:
        primary = "Soft / limited upside"
    elif extracted_regime:
        primary = extracted_regime
    else:
        primary = derived_regime or "N/A"

    if primary:
        primary = re.sub(r"^\w", lambda match: match.group(0).upper(), primary)
    helper = (
        _optional_text(selected["macro_summary"])
        or _optional_text(selected["key_driver"])
        or "No regime context extracted."
    )
    return primary, helper


def _build_key_driver_display(selected: pd.Series) -> tuple[str, str]:
    primary = _text_value(selected["key_driver"], "N/A")
    helper = "Main market-moving catalyst for the selected month."
    return primary, helper


def _render_supply_section(selected: pd.Series) -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        _section_card(
            "Brazil",
            "<br>".join(
                [
                    f"<strong>Cane crush:</strong> {_format_number(selected['brazil_cane_crush_mmt'], ' mmt')}",
                    f"<strong>Sugar production:</strong> {_format_number(selected['brazil_sugar_production_mmt'], ' mmt')}",
                    f"<strong>Sugar mix:</strong> {_format_number(selected['brazil_sugar_mix_pct'], '%')}",
                    f"<strong>Note:</strong> {_text_value(selected['brazil_note'], 'No Brazil-specific note extracted.')}",
                ]
            ),
        )
    with col2:
        _section_card(
            "India",
            "<br>".join(
                [
                    f"<strong>Current production:</strong> {_format_number(selected['india_current_production_mmt'], ' mmt')}",
                    f"<strong>Final outlook:</strong> {_format_number(selected['india_final_outlook_mmt'], ' mmt')}",
                    f"<strong>Exports:</strong> {_text_value(selected['india_exports_note'], 'No India-specific export note extracted.')}",
                    f"<strong>Note:</strong> {_text_value(selected['india_note'], 'No India-specific note extracted.')}",
                ]
            ),
        )
    with col3:
        _section_card(
            "Thailand",
            "<br>".join(
                [
                    f"<strong>Production outlook:</strong> {_format_number(selected['thailand_production_outlook_mmt'], ' mmt')}",
                    f"<strong>Ethanol diversion:</strong> {_format_number(selected['thailand_ethanol_diversion_kmt'], ' kmt')}",
                    f"<strong>Note:</strong> {_text_value(selected['thailand_note'], 'No Thailand-specific note extracted.')}",
                ]
            ),
        )


def _render_trade_section(selected: pd.Series) -> None:
    col1, col2 = st.columns(2)
    with col1:
        _section_card(
            "Trade / Risk",
            "<br>".join(
                [
                    f"<strong>Major disruption:</strong> {_text_value(selected['major_trade_disruption'], 'None highlighted.')}",
                    f"<strong>Trade summary:</strong> {_text_value(selected['trade_summary'], 'No trade summary extracted.')}",
                    f"<strong>Positioning:</strong> {_text_value(selected['market_positioning_note'], 'No positioning note extracted.')}",
                ]
            ),
        )
    with col2:
        _section_card(
            "Market Tone",
            "<br>".join(
                [
                    f"<strong>Key driver:</strong> {_text_value(selected['key_driver'], 'No key driver extracted.')}",
                    f"<strong>Market regime:</strong> {_text_value(selected['market_regime'], 'No regime extracted.')}",
                    f"<strong>Why traders care:</strong> {_text_value(selected['macro_summary'], 'No concise market framing extracted.')}",
                ]
            ),
        )


def _render_report_rag_demo(reports: list) -> None:
    st.markdown("### Report Q&A")
    st.caption(
        "Ask grounded questions across the loaded ED&F Man sugar reports. Retrieval uses a PageIndex-style report/page tree before reading source text."
    )

    st.markdown("**Suggestions**")
    suggestion_cols = st.columns(2)
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        with suggestion_cols[index % 2]:
            if st.button(suggestion, key=f"rag_suggestion_{index}", width="stretch"):
                st.session_state["report_rag_question"] = suggestion

    default_question = SUGGESTED_QUESTIONS[0]
    if "report_rag_question" not in st.session_state:
        st.session_state["report_rag_question"] = default_question

    question = st.text_input("Ask a question", key="report_rag_question")
    result = answer_report_question(question, reports)

    answer_tab, evidence_tab = st.tabs(["Answer", "Evidence"])

    with answer_tab:
        _section_card("Answer", result.answer.replace("\n", "<br>"))

    with evidence_tab:
        evidence_rows = [
            {
                "Source": item.record.source_type,
                "Title": item.record.title,
                "Month": item.record.month,
                "Page": item.record.page_number or "Summary",
                "Retrieval score": item.retrieval_score,
                "Rerank score": item.rerank_score,
                "Matched terms": ", ".join(item.matched_terms),
                "Search path": item.search_path,
                "Reasoning": item.reasoning,
                "Citation": item.record.citation,
            }
            for item in result.evidence
        ]
        if evidence_rows:
            st.dataframe(evidence_rows, width="stretch", hide_index=True)
        else:
            st.info("No report evidence was retrieved for this question.")

        for item in result.evidence:
            with st.expander(f"{item.record.source_type}: {item.record.title}"):
                st.write(item.record.text)
                st.caption(item.record.citation)


def _render_evidence_panel(selected: pd.Series, show_raw_evidence: bool) -> None:
    if not show_raw_evidence:
        return

    with st.expander("Evidence / Transparency", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Extracted JSON**")
            raw_payload = selected.drop(labels=["source_snippets"]).to_dict()
            raw_payload["source_snippets"] = selected["source_snippets"]
            st.code(json.dumps(raw_payload, indent=2, default=str), language="json")
        with col2:
            st.markdown("**Source snippets**")
            st.json(selected["source_snippets"])
            st.markdown("**Extracted text preview**")
            st.text(_text_value(selected["extracted_text_preview"], "No preview available."))


def _render_dashboard_page(frame: pd.DataFrame, selected_month: str, show_raw_evidence: bool) -> None:
    display_frame = frame[frame["month"] == selected_month]
    selected = display_frame.iloc[-1]
    latest = latest_row(frame)

    st.markdown("### KPI Cards")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        _metric_card(
            "NY11 Price",
            _format_number(selected["ny11_front_month_price"], " c/lb"),
            f"Latest available month: {latest.month if latest else selected['month']}",
        )
    with kpi2:
        _metric_card(
            "MoM Change",
            _format_change(selected["ny11_mom_change_pct"]),
            "Based on extracted NY11 front-month values.",
        )
    with kpi3:
        regime_value, regime_help = _build_market_regime_display(selected)
        _metric_card(
            "Market Regime",
            regime_value,
            regime_help,
        )
    with kpi4:
        key_driver_value, key_driver_help = _build_key_driver_display(selected)
        _metric_card(
            "Key Driver",
            key_driver_value,
            f"{key_driver_help} Selected month: {selected['month']}",
        )

    st.markdown("### Price Trend")
    st.altair_chart(_build_trend_chart(frame), width="stretch")
    caption = (
        "Blue shows NY11 in cents per pound, while the dashed amber line shows Brent in dollars per barrel so you can compare direction rather than absolute level."
    )
    st.caption(caption)

    st.markdown("### Supply Drivers")
    _render_supply_section(selected)

    st.markdown("### Trade / Risk")
    _render_trade_section(selected)

    _render_evidence_panel(selected, show_raw_evidence)


def _render_ask_question_page(reports: list) -> None:
    _render_report_rag_demo(reports)


def run_app() -> None:
    st.set_page_config(
        page_title="Global Sugar Market Insights Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    _inject_styles()

    page = st.sidebar.radio("Page", ["Dashboard", "Ask a Question"], index=0)

    st.markdown(
        """
        <div class="hero">
            <h1>Global Sugar Market Insights</h1>
            <p>AI-assisted extraction, dashboarding, and report Q&A for monthly sugar reports</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_col1, top_col2, top_col3 = st.columns([1.4, 1, 1])

    reports: list = []
    force_reextract = False
    with top_col2:
        force_reextract = st.button("Re-extract reports", width="stretch")
    with top_col3:
        show_raw_evidence = st.toggle("Show raw evidence", value=False)

    try:
        reports = load_reports(force_reextract=force_reextract)
    except Exception as exc:
        st.error(f"Unable to load reports: {exc}")
        st.stop()

    frame = reports_to_dataframe(reports)
    if frame.empty:
        st.warning("No reports found in the reports directory.")
        st.stop()

    if page == "Dashboard":
        month_options = frame["month"].tolist()
        selected_month = st.selectbox("Month selector", month_options, index=len(month_options) - 1)
        _render_dashboard_page(frame, selected_month, show_raw_evidence)
    else:
        _render_ask_question_page(reports)
