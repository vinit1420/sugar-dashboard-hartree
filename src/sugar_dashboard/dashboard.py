from __future__ import annotations

import json
import re

import altair as alt
import pandas as pd
import streamlit as st

from sugar_dashboard.pipeline import latest_row, load_reports, reports_to_dataframe
from sugar_dashboard.rag_workflow import generate_margin_answer


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
    return f"{value:,.{decimals}f}{suffix}"


def _format_change(value: float | None) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


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
    extracted_regime = selected["market_regime"]
    derived_regime = selected["regime_label"]
    context = " ".join(
        part for part in [selected["macro_summary"], selected["key_driver"], selected["trade_summary"]] if part
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
    helper = selected["macro_summary"] or selected["key_driver"] or "No regime context extracted."
    return primary, helper


def _build_key_driver_display(selected: pd.Series) -> tuple[str, str]:
    primary = selected["key_driver"] or "N/A"
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
                    f"<strong>Note:</strong> {selected['brazil_note'] or 'No Brazil-specific note extracted.'}",
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
                    f"<strong>Exports:</strong> {selected['india_exports_note'] or 'No India-specific export note extracted.'}",
                    f"<strong>Note:</strong> {selected['india_note'] or 'No India-specific note extracted.'}",
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
                    f"<strong>Note:</strong> {selected['thailand_note'] or 'No Thailand-specific note extracted.'}",
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
                    f"<strong>Major disruption:</strong> {selected['major_trade_disruption'] or 'None highlighted.'}",
                    f"<strong>Trade summary:</strong> {selected['trade_summary'] or 'No trade summary extracted.'}",
                    f"<strong>Positioning:</strong> {selected['market_positioning_note'] or 'No positioning note extracted.'}",
                ]
            ),
        )
    with col2:
        _section_card(
            "Market Tone",
            "<br>".join(
                [
                    f"<strong>Key driver:</strong> {selected['key_driver'] or 'No key driver extracted.'}",
                    f"<strong>Market regime:</strong> {selected['market_regime'] or 'No regime extracted.'}",
                    f"<strong>Why traders care:</strong> {selected['macro_summary'] or 'No concise market framing extracted.'}",
                ]
            ),
        )


def _render_summary_panel(selected: pd.Series) -> None:
    st.markdown("### AI Summary Panel")
    _section_card("Executive Summary", selected["executive_summary"] or "No executive summary extracted.")
    col1, col2 = st.columns(2)
    with col1:
        bullets = selected["what_changed"] or []
        formatted = "".join(f"<li>{item}</li>" for item in bullets) or "<li>No changes extracted.</li>"
        _section_card("What Changed", f"<ul class='bullet-list'>{formatted}</ul>")
    with col2:
        bullets = selected["why_it_matters"] or []
        formatted = "".join(f"<li>{item}</li>" for item in bullets) or "<li>No impact notes extracted.</li>"
        _section_card("Why It Matters", f"<ul class='bullet-list'>{formatted}</ul>")


def _render_margin_rag_demo() -> None:
    st.markdown("### Liquid Feed Margin RAG Workflow")
    st.caption(
        "A prototype workflow for the interview use case: grounded margin explanation across ERP, procurement, logistics, and market commentary."
    )

    default_question = "Why did liquid feed margin tighten in the Gulf region last month?"
    question = st.text_input("Commercial analyst question", value=default_question)
    result = generate_margin_answer(question)

    workflow_tab, answer_tab, evidence_tab = st.tabs(["Workflow", "Grounded Answer", "Retrieved Evidence"])

    with workflow_tab:
        for index, step in enumerate(result.workflow_steps, start=1):
            st.markdown(f"**{index}. {step}**")

    with answer_tab:
        _section_card("Question", result.question)
        _section_card("Generated Explanation", result.answer)
        _section_card("Confidence", result.confidence)

    with evidence_tab:
        evidence_rows = [
            {
                "Source": item.record.source_type,
                "Title": item.record.title,
                "Region": item.record.region,
                "Period": item.record.period,
                "Retrieval score": item.retrieval_score,
                "Rerank score": item.rerank_score,
                "Matched terms": ", ".join(item.matched_terms),
                "Citation": item.record.citation,
            }
            for item in result.evidence
        ]
        st.dataframe(evidence_rows, use_container_width=True, hide_index=True)

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
            st.text(selected["extracted_text_preview"] or "No preview available.")


def run_app() -> None:
    st.set_page_config(
        page_title="Global Sugar Market Insights Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Global Sugar Market Insights Dashboard</h1>
            <p>AI-assisted extraction and summarization of monthly sugar reports</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_col1, top_col2, top_col3 = st.columns([1.4, 1, 1])
    with top_col1:
        st.caption("A lightweight internal commodity intelligence tool for Jan-Mar 2026 market dynamics.")

    reports: list = []
    force_reextract = False
    with top_col2:
        force_reextract = st.button("Re-extract reports", use_container_width=True)
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

    month_options = ["All"] + frame["month"].tolist()
    selected_month = st.selectbox("Month selector", month_options, index=0)

    display_frame = frame if selected_month == "All" else frame[frame["month"] == selected_month]
    selected = display_frame.iloc[-1]
    latest = latest_row(frame)

    st.markdown("### KPI Cards")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        _metric_card(
            "Latest NY11 Price" if selected_month == "All" else "NY11 Price",
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
    st.altair_chart(_build_trend_chart(frame), use_container_width=True)
    caption = (
        "Blue shows NY11 in cents per pound, while the dashed amber line shows Brent in dollars per barrel so you can compare direction rather than absolute level."
    )
    st.caption(caption)

    st.markdown("### Supply Drivers")
    _render_supply_section(selected)

    st.markdown("### Trade / Risk")
    _render_trade_section(selected)

    _render_margin_rag_demo()

    _render_summary_panel(selected)
    _render_evidence_panel(selected, show_raw_evidence)
