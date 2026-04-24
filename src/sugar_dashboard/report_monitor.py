from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from sugar_dashboard.config import REPORT_SOURCE_URL, REPORTS_DIR


USER_AGENT = "sugar-dashboard-report-monitor/1.0"


@dataclass(frozen=True)
class ReportLink:
    title: str
    url: str

    @property
    def file_name(self) -> str:
        path_name = Path(unquote(urlparse(self.url).path)).name
        if path_name.lower().endswith(".pdf"):
            return path_name
        cleaned = re.sub(r"[^A-Za-z0-9]+", "-", self.title).strip("-")
        return f"{cleaned}.pdf"


@dataclass(frozen=True)
class DownloadedReport:
    title: str
    url: str
    path: Path


class SugarReportLinkParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self.base_url = base_url
        self._current_href: str | None = None
        self._current_text: list[str] = []
        self.links: list[ReportLink] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self._current_href = href
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._current_href:
            return

        title = " ".join(part.strip() for part in self._current_text if part.strip())
        href = self._current_href
        self._current_href = None
        self._current_text = []

        if "monthly sugar note" not in title.lower():
            return

        self.links.append(ReportLink(title=title, url=urljoin(self.base_url, href)))


def _fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read()
    except HTTPError as exc:
        raise RuntimeError(f"Unable to fetch {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to fetch {url}: {exc.reason}") from exc


def discover_report_links(source_url: str = REPORT_SOURCE_URL) -> list[ReportLink]:
    html = _fetch_bytes(source_url).decode("utf-8", errors="replace")
    parser = SugarReportLinkParser(base_url=source_url)
    parser.feed(html)

    seen: set[str] = set()
    unique_links: list[ReportLink] = []
    for link in parser.links:
        if link.url in seen:
            continue
        seen.add(link.url)
        unique_links.append(link)
    return unique_links


def download_missing_reports(
    report_links: list[ReportLink] | None = None,
    reports_dir: Path = REPORTS_DIR,
) -> list[DownloadedReport]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    links = report_links if report_links is not None else discover_report_links()
    downloaded: list[DownloadedReport] = []

    for link in links:
        target_path = reports_dir / link.file_name
        if target_path.exists():
            continue

        payload = _fetch_bytes(link.url)
        if not payload.startswith(b"%PDF"):
            raise RuntimeError(f"Downloaded content for {link.title} did not look like a PDF: {link.url}")

        target_path.write_bytes(payload)
        downloaded.append(DownloadedReport(title=link.title, url=link.url, path=target_path))

    return downloaded
