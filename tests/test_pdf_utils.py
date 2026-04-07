"""Unit tests for pdf_utils — tasks 2.3 and 2.4."""

import io
import pytest
import pypdf

from src.pdf_utils import extract_pdf_pages, is_valid_pdf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf(num_pages: int = 1, page_text: str = "Hello") -> bytes:
    """Create a minimal in-memory PDF with *num_pages* pages."""
    writer = pypdf.PdfWriter()
    for _ in range(num_pages):
        page = pypdf.PageObject.create_blank_page(width=200, height=200)
        writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Task 2.3 — is_valid_pdf
# ---------------------------------------------------------------------------

def test_non_pdf_bytes_rejected():
    """Non-PDF bytes must return False."""
    assert is_valid_pdf(b"not a pdf at all") is False


def test_empty_bytes_rejected():
    assert is_valid_pdf(b"") is False


def test_wrong_magic_bytes_rejected():
    assert is_valid_pdf(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100) is False


def test_truncated_pdf_rejected():
    """Bytes that start with %PDF- but are not a valid PDF must return False."""
    assert is_valid_pdf(b"%PDF-1.4 truncated garbage") is False


def test_valid_pdf_accepted():
    pdf_bytes = _make_pdf()
    assert is_valid_pdf(pdf_bytes) is True


# ---------------------------------------------------------------------------
# Task 2.4 — extract_pdf_pages
# ---------------------------------------------------------------------------

def test_single_page_pdf_returns_one_entry():
    pdf_bytes = _make_pdf(num_pages=1)
    pages = extract_pdf_pages(pdf_bytes)
    assert len(pages) == 1


def test_multi_page_pdf_returns_correct_count():
    pdf_bytes = _make_pdf(num_pages=3)
    pages = extract_pdf_pages(pdf_bytes)
    assert len(pages) == 3


def test_pages_are_strings():
    pdf_bytes = _make_pdf(num_pages=2)
    pages = extract_pdf_pages(pdf_bytes)
    assert all(isinstance(p, str) for p in pages)


def test_zero_page_pdf_raises_value_error():
    """A PDF with zero pages must raise ValueError."""
    writer = pypdf.PdfWriter()
    buf = io.BytesIO()
    writer.write(buf)
    zero_page_pdf = buf.getvalue()
    with pytest.raises(ValueError, match="zero pages"):
        extract_pdf_pages(zero_page_pdf)
