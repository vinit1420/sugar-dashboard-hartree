# Sugar Market Insights Dashboard

AI-assisted Streamlit dashboard for monthly sugar-market reports.

Live app:
[sugar-dashboard-hartree.streamlit.app](https://sugar-dashboard-hartree-96qrmwq8laz74qdx5qed75.streamlit.app/)

## Overview

This project ingests monthly sugar-market PDF reports, extracts structured signals with an LLM, stores the outputs as JSON, and renders a one-page dashboard for fast analyst review.

The app is designed as a lightweight commodity intelligence tool for summarizing:

- price moves in NY11
- supply-side developments in Brazil, India, and Thailand
- trade and risk signals
- concise AI-generated monthly summaries

## Features

- PDF ingestion from the local `reports/` folder
- Page-by-page text extraction with PyMuPDF
- Schema-based structured extraction using OpenAI
- Cached processed JSON outputs in `data/processed/`
- KPI cards, trend chart, supply cards, trade/risk panel, and AI summary panel
- Evidence view for transparency and auditability

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- reports/
|-- data/processed/
`-- src/sugar_dashboard/
    |-- config.py
    |-- dashboard.py
    |-- extractor.py
    |-- models.py
    |-- pdf_ingestion.py
    `-- pipeline.py
```

## How It Works

1. PDF reports are loaded from `reports/`.
2. Text is extracted page by page.
3. The combined report text is sent to OpenAI with a strict schema prompt.
4. Structured data is validated with Pydantic.
5. Derived metrics are computed and cached as JSON.
6. Streamlit renders the dashboard from cached outputs.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Add your OpenAI API key in `.env`:

```env
OPENAI_API_KEY=your_key_here
```

Start the app:

```bash
streamlit run app.py
```

## Deployment

This project is deployed on Streamlit Community Cloud.

Deployment settings:

- Repository: `vinit1420/sugar-dashboard-hartree`
- Branch: `main`
- Main file: `app.py`

Streamlit secret format:

```toml
OPENAI_API_KEY = "your_key_here"
```

## Notes

- The app can load from cached JSON files without re-extracting every time.
- Clicking `Re-extract reports` triggers fresh OpenAI extraction.
- `.env` is ignored and is not committed to GitHub.

