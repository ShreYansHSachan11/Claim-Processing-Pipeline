"""Tests for POST /api/process endpoint."""

import io
from unittest.mock import patch

import pypdf
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from src.main import app
from src.models import BillResult, ClaimResult, IdentityResult

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pdf() -> bytes:
    """Create a minimal valid single-page PDF in memory."""
    writer = pypdf.PdfWriter()
    writer.add_page(pypdf.PageObject.create_blank_page(width=200, height=200))
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _post(claim_id: str, file_bytes: bytes, filename: str = "test.pdf") -> object:
    return client.post(
        "/api/process",
        data={"claim_id": claim_id},
        files={"file": (filename, file_bytes, "application/pdf")},
    )


# ---------------------------------------------------------------------------
# Property test 7.6 — Property 10: missing/whitespace claim_id → 422
# Feature: claim-processing-pipeline, Property 10: missing claim_id rejected with 422
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------

@given(claim_id=st.one_of(st.just(""), st.text(alphabet=" \t\n", min_size=1)))
@settings(max_examples=50)
def test_empty_or_whitespace_claim_id_returns_422(claim_id: str) -> None:
    """Property 10: any empty or whitespace-only claim_id must be rejected with 422."""
    pdf_bytes = make_pdf()
    response = _post(claim_id=claim_id, file_bytes=pdf_bytes)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Property test 7.7 — Property 9: non-PDF file upload → 422
# Feature: claim-processing-pipeline, Property 9: non-PDF file upload returns 422
# Validates: Requirements 1.3
# ---------------------------------------------------------------------------

@given(data=st.binary(min_size=1).filter(lambda b: not b.startswith(b"%PDF-")))
@settings(max_examples=50)
def test_non_pdf_upload_returns_422(data: bytes) -> None:
    """Property 9: any non-PDF bytes must be rejected with 422."""
    response = _post(claim_id="CLM-001", file_bytes=data, filename="bad.bin")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Unit test 7.8 — valid request returns 200 with ClaimResult JSON shape
# Validates: Requirements 1.1, 1.2
# ---------------------------------------------------------------------------

def test_valid_request_returns_200_with_claim_result_shape() -> None:
    """A valid claim_id + valid PDF should return 200 with ClaimResult JSON."""
    mock_result = ClaimResult(
        claim_id="CLM-001",
        page_classification_map={"other": [0]},
        identity=None,
        discharge_summary=None,
        itemized_bill=None,
    )

    with patch("src.main.run_pipeline", return_value=mock_result):
        response = _post(claim_id="CLM-001", file_bytes=make_pdf())

    assert response.status_code == 200
    body = response.json()
    assert body["claim_id"] == "CLM-001"
    assert "page_classification_map" in body
    assert "identity" in body
    assert "discharge_summary" in body
    assert "itemized_bill" in body


# ---------------------------------------------------------------------------
# Unit test 7.9 — pipeline exception returns 500
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

def test_pipeline_exception_returns_500() -> None:
    """An unhandled exception from run_pipeline must produce HTTP 500."""
    with patch("src.main.run_pipeline", side_effect=RuntimeError("LLM unavailable")):
        response = _post(claim_id="CLM-002", file_bytes=make_pdf())

    assert response.status_code == 500
    body = response.json()
    assert "Pipeline error" in body.get("detail", "")
