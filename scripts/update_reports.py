from __future__ import annotations

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sugar_dashboard.report_monitor import discover_report_links, download_missing_reports
from sugar_dashboard.pipeline import load_reports


def main() -> int:
    report_links = discover_report_links()
    print(f"Discovered {len(report_links)} ED&F Man Monthly Sugar Note link(s).")

    downloaded = download_missing_reports(report_links=report_links)
    if not downloaded:
        print("No new reports found. Dashboard data is already up to date.")
        return 0

    for report in downloaded:
        print(f"Downloaded {report.title}: {report.path.name}")

    processed_reports = load_reports(force_reextract=False)
    print(f"Processed {len(processed_reports)} report(s) into dashboard-ready JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
