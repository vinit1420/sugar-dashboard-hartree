from __future__ import annotations

from textwrap import dedent

from openai import OpenAI

from sugar_dashboard.config import Settings
from sugar_dashboard.models import IngestedReport, MarketReportExtraction


SYSTEM_PROMPT = dedent(
    """
    You extract structured sugar-market intelligence from analyst PDF reports.

    Return a JSON object that matches the schema exactly.
    Rules:
    - Use only evidence from the supplied report text.
    - If a field is missing or unsupported, return null.
    - Do not infer values that are not explicitly supported.
    - Keep summaries concise, analyst-style, and business-facing.
    - "what_changed" and "why_it_matters" should each contain exactly 3 concise bullet strings when supported.
    - "source_snippets" must contain short exact supporting snippets for the listed fields when available, otherwise null.
    """
).strip()


class OpenAIReportExtractor:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for extraction when cached data is unavailable.")
        self.settings = settings
        self.client = OpenAI(api_key=settings.openai_api_key)

    def extract(self, report: IngestedReport) -> MarketReportExtraction:
        report_text = report.combined_text[: self.settings.extraction_max_characters]
        user_prompt = dedent(
            f"""
            Extract the monthly sugar market report into the required schema.

            Report file: {report.file_name}

            Report text:
            {report_text}
            """
        ).strip()

        response = self.client.responses.parse(
            model=self.settings.openai_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text_format=MarketReportExtraction,
        )

        if getattr(response, "output_parsed", None) is None:
            raise RuntimeError("OpenAI response did not contain a parsed structured payload.")

        return response.output_parsed
