from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT_DIR / "reports"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_SOURCE_URL = "https://www.edfmansugar.com/sugar-reports/"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = "gpt-4o-mini"
    extraction_max_characters: int = 50_000


load_dotenv(ROOT_DIR / ".env")


def get_settings() -> Settings:
    return Settings()
