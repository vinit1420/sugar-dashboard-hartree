# Project Spec: Sugar Market Insights Dashboard

## Objective
Build a Streamlit app that ingests monthly sugar-market PDF reports, extracts structured market signals with an LLM, stores them as JSON, and renders a one-page dashboard summarizing Jan–Mar 2026 market dynamics.

This is a showcase project for:
- AI-assisted document understanding
- structured signal extraction from unstructured PDFs
- business-facing market intelligence dashboards
- concise insight generation

The app should feel like a lightweight internal commodity intelligence tool.

---

## Core Product Story
Raw monthly sugar reports are difficult to scan quickly.

This app should:
1. read PDF reports
2. extract key numerical and narrative signals into structured JSON
3. generate concise monthly summaries
4. visualize key market movements in a single Streamlit dashboard

The output should be polished enough to demo in an interview.

---

## Primary User
A business or market analyst who wants:
- a fast summary of what changed month to month
- the key supply / macro / trade drivers
- a compact market dashboard instead of reading full reports

---

## Input Files
The app should support local PDFs in a `reports/` folder.

Initial files:
- `Monthly-Sugar-Note-Jan-2026.pdf`
- `Monthly-Sugar-Note-Feb-2026.pdf`
- `Monthly-Sugar-Note-Mar-2026.pdf`

---

## Functional Requirements

### 1. PDF ingestion
- Read all PDF files from `reports/`
- Extract text page by page
- Preserve page boundaries if possible
- Handle parsing errors gracefully

### 2. LLM-based structured extraction
For each report, extract a fixed schema in JSON.
Use a strict schema prompt.
If a field is missing, return `null`.
Do not infer unsupported values.

### 3. Derived metrics
After extraction, compute:
- month-over-month NY11 price change where possible
- directional flags for major metrics
- regime labels if not extracted directly
- simple trend-ready numeric fields

### 4. Streamlit dashboard
Single-page app with:
- header
- KPI cards
- trend chart
- supply drivers section
- macro drivers section
- trade/risk section
- monthly AI summaries
- expandable raw extracted JSON or evidence panel

### 5. Local caching / persistence
Store extracted structured outputs in `data/processed/` as JSON files.
App should load cached JSON if it exists.
Add a “Re-extract reports” button to rerun extraction.

---

## Non-Functional Requirements
- Clean, minimal UI
- Fast local startup
- Easy to explain in an interview
- Modular code
- Robust error handling
- No overengineering

---

## Suggested Tech Stack
- Python 3.11+
- Streamlit
- pandas
- plotly or altair
- pydantic for schema validation
- pdf text extraction library:
  - prefer `pymupdf` or `pdfplumber`
- OpenAI API for extraction/summarization
- dotenv for environment management

---

## App Layout

### Header
Title:
`Global Sugar Market Insights Dashboard`

Subtitle:
`AI-assisted extraction and summarization of monthly sugar reports`

Controls:
- month selector (All / Jan / Feb / Mar)
- button: `Re-extract reports`
- toggle: `Show raw evidence`

---

### Section 1: KPI Cards
Display 4 cards:
1. Latest NY11 price
2. MoM change
3. Market regime
4. Key driver of selected month

If "All" is selected:
- show latest available month
- label clearly

---

### Section 2: Price Trend
Line chart:
- x-axis = month
- y-axis = NY11 front-month price

Optional:
- secondary line for Brent oil

Caption:
- concise explanation of Jan → Feb → Mar narrative

---

### Section 3: Supply Drivers
Three columns or grouped cards:

#### Brazil
- cane crush (mmt)
- sugar production (mmt)
- sugar mix (%)
- narrative note

#### India
- current production (mmt)
- final outlook (mmt)
- exports note
- narrative note

#### Thailand
- production outlook (mmt)
- ethanol diversion if available
- narrative note

---

### Section 4: Macro Drivers
Show:
- Brent oil
- fertilizer / energy cost note
- logistics or freight disruption note
- key macro summary

Could be:
- KPI cards + bullet summary
or
- one compact chart + text panel

---

### Section 5: Trade / Risk
Show:
- major trade disruption
- export / import issue
- geopolitical risk note
- market sentiment / positioning

---

### Section 6: AI Summary Panel
For selected month:
- one-paragraph executive summary
- 3 bullet “what changed”
- 3 bullet “why it matters”

This should read like a concise analyst note.

---

### Section 7: Evidence / Transparency
Expandable section:
- extracted JSON
- source snippets
- extracted text preview

Purpose:
show auditability and reduce “black box” feel

---

## Extraction Schema

Use a strict schema like this:

```json
{
  "month": "Jan 2026",
  "report_date": "2026-01-28",
  "ny11_front_month_price": 14.71,
  "ny11_price_change_pct": -2.0,
  "london5_front_month_price": 412.2,
  "brent_oil": 68.4,
  "market_regime": "Soft / limited upside",
  "key_driver": "Comfortable global trade flows with strong Brazil exports",
  "brazil_cane_crush_mmt": 600.0,
  "brazil_sugar_production_mmt": 40.2,
  "brazil_sugar_mix_pct": 50.8,
  "india_current_production_mmt": 15.9,
  "india_final_outlook_mmt": 30.0,
  "thailand_production_outlook_mmt": 10.5,
  "thailand_ethanol_diversion_kmt": 671.0,
  "major_trade_disruption": null,
  "market_positioning_note": null,
  "macro_summary": "Trade-war risk and volatile macro backdrop.",
  "supply_summary": "Brazil stable; India and Thailand mixed.",
  "trade_summary": "Global flows remain comfortable.",
  "executive_summary": "One concise paragraph.",
  "what_changed": [
    "string",
    "string",
    "string"
  ],
  "why_it_matters": [
    "string",
    "string",
    "string"
  ],
  "source_snippets": {
    "ny11_front_month_price": "exact supporting text snippet",
    "brazil_sugar_production_mmt": "exact supporting text snippet",
    "india_final_outlook_mmt": "exact supporting text snippet"
  }
}