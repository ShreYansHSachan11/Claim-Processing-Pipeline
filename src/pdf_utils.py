"""PDF validation and page-to-image conversion utilities."""

import base64
from io import BytesIO

import fitz  # pymupdf
from pypdf import PdfReader

_PDF_MAGIC = b"%PDF-"


def is_valid_pdf(data: bytes) -> bool:
    """Return True if *data* starts with the PDF magic bytes and can be parsed."""
    if not data or not data.startswith(_PDF_MAGIC):
        return False
    try:
        reader = PdfReader(BytesIO(data))
        _ = len(reader.pages)
        return True
    except Exception:
        return False


def extract_pdf_pages(data: bytes) -> list[str]:
    """Convert each PDF page to a base64-encoded PNG image string.

    Returns a list of base64 strings (one per page) suitable for GPT-4o vision.

    Raises:
        ValueError: if the PDF contains zero pages.
    """
    doc = fitz.open(stream=data, filetype="pdf")
    if doc.page_count == 0:
        raise ValueError("PDF contains zero pages")

    pages = []
    for page in doc:
        # Render at 150 DPI — good balance of quality vs token cost
        mat = fitz.Matrix(100 / 72, 100 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        pages.append(b64)

    return pages
