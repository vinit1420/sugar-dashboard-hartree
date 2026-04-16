from __future__ import annotations

from pathlib import Path

from sugar_dashboard.models import IngestedReport, PDFPage


def _extract_with_pymupdf(pdf_path: Path) -> list[PDFPage]:
    import fitz

    pages: list[PDFPage] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            pages.append(PDFPage(page_number=index, text=page.get_text("text")))
    return pages


def _extract_with_pdfplumber(pdf_path: Path) -> list[PDFPage]:
    import pdfplumber

    pages: list[PDFPage] = []
    with pdfplumber.open(pdf_path) as document:
        for index, page in enumerate(document.pages, start=1):
            pages.append(PDFPage(page_number=index, text=page.extract_text() or ""))
    return pages


def extract_pdf_pages(pdf_path: Path) -> IngestedReport:
    errors: list[str] = []
    pages: list[PDFPage] | None = None

    try:
        pages = _extract_with_pymupdf(pdf_path)
    except ImportError as exc:
        errors.append(f"PyMuPDF unavailable: {exc}")
    except Exception as exc:
        errors.append(f"PyMuPDF failed for {pdf_path.name}: {exc}")

    if pages is None:
        try:
            pages = _extract_with_pdfplumber(pdf_path)
        except ImportError as exc:
            errors.append(f"pdfplumber unavailable: {exc}")
        except Exception as exc:
            errors.append(f"pdfplumber failed for {pdf_path.name}: {exc}")

    if pages is None:
        joined_errors = "; ".join(errors) or "No PDF extraction backend succeeded."
        raise RuntimeError(joined_errors)

    return IngestedReport(
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        pages=pages,
    )
