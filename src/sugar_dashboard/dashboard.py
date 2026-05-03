from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from sugar_dashboard.market_data import MarketSeries, build_market_explorer_frame, fetch_market_history
from sugar_dashboard.pipeline import latest_row, load_reports, reports_to_dataframe
from sugar_dashboard.rag_workflow import SUGGESTED_QUESTIONS, RagAnswer, answer_report_question


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


def _selected_context(selected: pd.Series) -> str:
    parts = [
        _optional_text(selected.get("market_regime")),
        _optional_text(selected.get("key_driver")),
        _optional_text(selected.get("macro_summary")),
        _optional_text(selected.get("supply_summary")),
        _optional_text(selected.get("trade_summary")),
        _optional_text(selected.get("major_trade_disruption")),
        _optional_text(selected.get("market_positioning_note")),
        _optional_text(selected.get("extracted_text_preview")),
    ]
    parts.extend(_list_value(selected.get("what_changed")))
    parts.extend(_list_value(selected.get("why_it_matters")))
    return " ".join(part for part in parts if part).lower()


def _derive_regime_from_context(context: str) -> str | None:
    bullish_terms = (
        "tightening supply",
        "tighten",
        "tight",
        "support prices",
        "supportive",
        "bullish",
        "rally",
        "short covering",
        "higher ethanol",
        "upward price momentum",
    )
    bearish_terms = (
        "comfortable surplus",
        "bearish",
        "declined",
        "decreased",
        "lower prices",
        "short positions",
        "limiting any upward",
        "stronger than expected",
        "surplus",
    )
    bullish_score = sum(1 for term in bullish_terms if term in context)
    bearish_score = sum(1 for term in bearish_terms if term in context)
    if bearish_score > bullish_score:
        return "Bearish / surplus pressure"
    if bullish_score > bearish_score:
        return "Supportive / upside risk"
    if bullish_score or bearish_score:
        return "Mixed / event-driven"
    return None


def _derive_key_driver_from_context(selected: pd.Series) -> str | None:
    explicit = _optional_text(selected.get("key_driver"))
    if explicit:
        return explicit

    context = _selected_context(selected)
    if "comfortable surplus" in context and "thailand" in context:
        return "Stronger Thailand/China output and rebuilt short positioning pressured NY11."
    if "middle east" in context and "oil" in context:
        return "Middle East conflict and oil/ethanol linkage drove sugar risk."
    if "ethanol" in context and "sugar mix" in context:
        return "Brazil ethanol parity and sugar-mix uncertainty are the key watch points."

    what_changed = _list_value(selected.get("what_changed"))
    if what_changed:
        return what_changed[0]
    why_it_matters = _list_value(selected.get("why_it_matters"))
    if why_it_matters:
        return why_it_matters[0]
    return None


def _derive_market_summary_from_context(selected: pd.Series) -> str | None:
    explicit = _optional_text(selected.get("macro_summary"))
    if explicit:
        return explicit

    context = _selected_context(selected)
    if "comfortable surplus" in context:
        return "Global balance moved toward surplus as stronger Thailand and China output offset weaker areas."
    if "large net short" in context or "short positions" in context:
        return "Speculative shorts are limiting upside but could become a covering catalyst."
    why_it_matters = _list_value(selected.get("why_it_matters"))
    if why_it_matters:
        return why_it_matters[0]
    return None


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


@st.cache_data(ttl=60 * 60, show_spinner=False)
def _load_market_history(years: int = 2) -> list[MarketSeries]:
    return fetch_market_history(years=years)


def _build_plotly_market_chart(plot_frame: pd.DataFrame, selected_lines: list[str], transform_mode: str):
    import plotly.graph_objects as go

    y_column = {
        "Raw prices": "close",
        "Indexed to 100": "indexed",
        "Daily % change": "mom_change",
    }[transform_mode]
    y_title = {
        "Raw prices": "Price",
        "Indexed to 100": "Index",
        "Daily % change": "Daily change (%)",
    }[transform_mode]

    figure = go.Figure()
    for label in selected_lines:
        line_frame = plot_frame[plot_frame["label"] == label].dropna(subset=[y_column])
        if line_frame.empty:
            continue
        dash = "dash" if label.endswith("LY") else "solid"
        figure.add_trace(
            go.Scatter(
                x=line_frame["display_date"],
                y=line_frame[y_column],
                mode="lines",
                name=label,
                line={"width": 2.5, "dash": dash},
                customdata=line_frame[["date", "unit", "close"]],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Display date: %{x|%b %d, %Y}<br>"
                    "Source date: %{customdata[0]|%b %d, %Y}<br>"
                    "Value: %{y:.2f}<br>"
                    "Raw close: %{customdata[2]:.2f} %{customdata[1]}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        height=460,
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        xaxis={"title": "Date", "rangeslider": {"visible": True}},
        yaxis={"title": y_title},
        template="plotly_white",
    )
    return figure


def _render_market_explorer() -> None:
    st.markdown("### Interactive Market Structure Explorer")
    st.caption(
        "Daily Yahoo Finance history for NY11 sugar and Brent, with last-year overlays shifted onto the current calendar for seasonal comparison."
    )

    market_series = _load_market_history(years=2)
    errors = [series.error for series in market_series if series.error]
    plot_frame = build_market_explorer_frame(market_series)
    if plot_frame.empty:
        st.info("Market history is unavailable right now. The report-based dashboard below is still available.")
        if errors:
            with st.expander("Market data diagnostics", expanded=False):
                for error in errors:
                    st.write(f"- {error}")
        return

    available_lines = sorted(plot_frame["label"].dropna().unique().tolist())
    default_lines = [line for line in available_lines if "NY11" in line or "Brent" in line][:4]

    control_col1, control_col2 = st.columns([1.5, 1])
    with control_col1:
        selected_lines = st.multiselect(
            "Lines",
            available_lines,
            default=default_lines,
            max_selections=6,
            help="Select up to six traces. LY lines are last year's prices shifted forward one calendar year.",
        )
    with control_col2:
        transform_mode = st.segmented_control(
            "View",
            ["Raw prices", "Indexed to 100", "Daily % change"],
            default="Indexed to 100",
        )

    if not selected_lines:
        st.info("Select at least one line to render the market explorer.")
        return

    figure = _build_plotly_market_chart(plot_frame, selected_lines, transform_mode)
    st.plotly_chart(figure, width="stretch")

    sources = ", ".join(f"{series.label}: {series.source}" for series in market_series if not series.frame.empty)
    st.caption(f"Source: {sources}. Data is delayed and should be treated as decision support, not official settlement data.")


def _build_market_regime_display(selected: pd.Series) -> tuple[str, str]:
    extracted_regime = _optional_text(selected["market_regime"])
    derived_regime = _optional_text(selected["regime_label"])
    context = _selected_context(selected)

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
        primary = derived_regime or _derive_regime_from_context(context) or "N/A"

    if primary:
        primary = re.sub(r"^\w", lambda match: match.group(0).upper(), primary)
    helper = (
        _optional_text(selected["macro_summary"])
        or _derive_key_driver_from_context(selected)
        or "No regime context extracted."
    )
    return primary, helper


def _build_key_driver_display(selected: pd.Series) -> tuple[str, str]:
    primary = _derive_key_driver_from_context(selected) or "N/A"
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
        key_driver_value, _ = _build_key_driver_display(selected)
        regime_value, _ = _build_market_regime_display(selected)
        market_summary = _derive_market_summary_from_context(selected)
        _section_card(
            "Market Tone",
            "<br>".join(
                [
                    f"<strong>Key driver:</strong> {key_driver_value if key_driver_value != 'N/A' else 'No key driver extracted.'}",
                    f"<strong>Market regime:</strong> {regime_value if regime_value != 'N/A' else 'No regime extracted.'}",
                    f"<strong>Why traders care:</strong> {market_summary or 'No concise market framing extracted.'}",
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
                st.session_state["report_rag_run_requested"] = True

    default_question = SUGGESTED_QUESTIONS[0]
    if "report_rag_question" not in st.session_state:
        st.session_state["report_rag_question"] = default_question

    with st.form("report_rag_form"):
        question = st.text_input("Ask a question", key="report_rag_question")
        submitted = st.form_submit_button("Run retrieval", width="stretch")

    should_run = submitted or st.session_state.pop("report_rag_run_requested", False)
    if should_run:
        status = st.status("Thinking through the loaded reports...", expanded=True)
        status.write("Building the report context.")
        status.write("Retrieving the most relevant evidence sections.")
        status.write("Drafting a grounded answer and checking support.")
        with st.spinner("Generating answer..."):
            st.session_state["report_rag_last_result"] = answer_report_question(question, reports)
        status.update(label="Answer ready", state="complete", expanded=False)

    result = st.session_state.get("report_rag_last_result")
    if result is None:
        st.info("Choose a suggested question or type one and click Run retrieval.")
        return

    answer_tab, evidence_tab = st.tabs(["Answer", "Evidence"])

    with answer_tab:
        _section_card("Answer", result.answer.replace("\n", "<br>"))

    with evidence_tab:
        _render_retrieval_tree(result)


def _render_retrieval_tree(result: RagAnswer) -> None:
    if not result.evidence:
        st.info("No report evidence was retrieved for this question.")
        return

    grouped_evidence: dict[tuple[str, str], list] = defaultdict(list)
    for item in result.evidence:
        report_file = item.record.citation.split(",", 1)[0]
        grouped_evidence[(item.record.month, report_file)].append(item)

    st.markdown("**Retrieval tree**")
    for (month, report_file), items in grouped_evidence.items():
        st.markdown(f"- **{month}** · `{report_file}`")
        for item in items:
            page_label = f"page {item.record.page_number}" if item.record.page_number else "extracted summary"
            section_label = item.record.title if item.record.source_type == "PageIndex section search" else page_label
            path = item.search_path or f"{month} > {page_label}"
            reason = item.reasoning or "Selected because it matched the question context."
            terms = ", ".join(item.matched_terms) if item.matched_terms else "tree reasoning"
            st.markdown(
                "\n".join(
                    [
                        f"  - **Section:** {section_label}",
                        f"    - Page: {page_label}",
                        f"    - Path: `{path}`",
                        f"    - Why selected: {reason}",
                        f"    - Signal: {terms}",
                        f"    - Citation: {item.record.citation}",
                    ]
                )
            )


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

    _render_market_explorer()

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
