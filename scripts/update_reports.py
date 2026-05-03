from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sugar_dashboard.report_monitor import discover_report_links, download_missing_reports
from sugar_dashboard.config import PROCESSED_DIR, get_settings
from sugar_dashboard.pipeline import load_reports


def _uncached_report_names() -> list[str]:
    report_paths = sorted((ROOT_DIR / "reports").glob("*.pdf"))
    return [
        report_path.name
        for report_path in report_paths
        if not (PROCESSED_DIR / f"{report_path.stem}.json").exists()
    ]


def main() -> int:
    report_links = discover_report_links()
    print(f"Discovered {len(report_links)} ED&F Man Monthly Sugar Note link(s).")

    downloaded = download_missing_reports(report_links=report_links)
    missing_cache_names = _uncached_report_names()
    settings = get_settings()

    for report in downloaded:
        print(f"Downloaded {report.title}: {report.path.name}")

    if not downloaded and not missing_cache_names:
        print("No new reports found. Dashboard data is already up to date.")
        return 0

    if missing_cache_names and not settings.openai_api_key:
        print(
            "OPENAI_API_KEY is not configured, so skipping extraction for uncached report(s): "
            + ", ".join(missing_cache_names)
        )
        print("Downloaded PDFs can still be committed; processed JSON will be generated when OPENAI_API_KEY is available.")
        return 0

    processed_reports = load_reports(force_reextract=False)
    print(f"Processed {len(processed_reports)} report(s) into dashboard-ready JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
