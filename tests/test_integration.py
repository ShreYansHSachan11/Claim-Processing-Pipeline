"""Integration tests for the full claim processing pipeline (tasks 8.1–8.3).

These tests exercise run_pipeline end-to-end using mocked LLM responses
and real PDF bytes created with pypdf.
"""
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from pypdf import PdfWriter

from src.models import (
    BillResult,
    ClaimResult,
    DischargeResult,
    IdentityResult,
    LineItem,
)
from src.nodes.segregator import PageClassification
from src.pdf_utils import extract_pdf_pages
from src.pipeline import run_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdf_bytes(num_pages: int, page_texts: list[str] | None = None) -> bytes:
    """Create a minimal valid PDF with *num_pages* pages using pypdf."""
    writer = PdfWriter()
    for i in range(num_pages):
        page = writer.add_blank_page(width=612, height=792)
        # pypdf blank pages have no text layer; that's fine — we only need
        # valid PDF bytes that extract_pdf_pages can parse.
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_mock_llm_full_pipeline() -> MagicMock:
    """
    Mock LLM for test 8.1: 3-page PDF.
    Segregator classifies: page 0 → identity_document, page 1 → discharge_summary,
    page 2 → itemized_bill.
    Extraction agents return populated results.
    """
    mock_llm = MagicMock()

    def structured_output_side_effect(schema):
        structured = MagicMock()
        if schema is PageClassification:
            structured.invoke.side_effect = [
                PageClassification(document_type="identity_document"),
                PageClassification(document_type="discharge_summary"),
                PageClassification(document_type="itemized_bill"),
            ]
        elif schema is IdentityResult:
            structured.invoke.return_value = IdentityResult(
                patient_name="Alice",
                date_of_birth="1990-01-01",
                id_numbers=["ID-123"],
                policy_details="Policy XYZ",
            )
        elif schema is DischargeResult:
            structured.invoke.return_value = DischargeResult(
                diagnosis="Flu",
                admission_date="2024-01-10",
                discharge_date="2024-01-15",
                attending_physician="Dr. Smith",
            )
        elif schema is BillResult:
            structured.invoke.return_value = BillResult(
                line_items=[LineItem(description="X-Ray", cost=150.0)],
                total_amount=150.0,
            )
        return structured

    mock_llm.with_structured_output.side_effect = structured_output_side_effect
    return mock_llm


def _make_mock_llm_no_identity() -> MagicMock:
    """
    Mock LLM for test 8.2: 2-page PDF.
    Segregator classifies: page 0 → discharge_summary, page 1 → itemized_bill.
    No identity pages.
    """
    mock_llm = MagicMock()

    def structured_output_side_effect(schema):
        structured = MagicMock()
        if schema is PageClassification:
            structured.invoke.side_effect = [
                PageClassification(document_type="discharge_summary"),
                PageClassification(document_type="itemized_bill"),
            ]
        elif schema is DischargeResult:
            structured.invoke.return_value = DischargeResult(diagnosis="Cold")
        elif schema is BillResult:
            structured.invoke.return_value = BillResult(
                line_items=[LineItem(description="Consultation", cost=80.0)],
                total_amount=80.0,
            )
        return structured

    mock_llm.with_structured_output.side_effect = structured_output_side_effect
    return mock_llm


def _make_mock_llm_no_bill() -> MagicMock:
    """
    Mock LLM for test 8.3: 2-page PDF.
    Segregator classifies: page 0 → identity_document, page 1 → discharge_summary.
    No itemized_bill pages.
    """
    mock_llm = MagicMock()

    def structured_output_side_effect(schema):
        structured = MagicMock()
        if schema is PageClassification:
            structured.invoke.side_effect = [
                PageClassification(document_type="identity_document"),
                PageClassification(document_type="discharge_summary"),
            ]
        elif schema is IdentityResult:
            structured.invoke.return_value = IdentityResult(patient_name="Bob")
        elif schema is DischargeResult:
            structured.invoke.return_value = DischargeResult(diagnosis="Sprain")
        return structured

    mock_llm.with_structured_output.side_effect = structured_output_side_effect
    return mock_llm


# ---------------------------------------------------------------------------
# Task 8.1 — full pipeline with multi-page PDF returns populated ClaimResult
# ---------------------------------------------------------------------------

def test_full_pipeline_multi_page_pdf_returns_populated_claim_result():
    """
    Integration test: a 3-page PDF with all three document types present
    should produce a fully populated ClaimResult.

    Validates: Requirements 1, 2, 3, 4, 5, 6, 7
    """
    pdf_bytes = _make_pdf_bytes(3)
    pdf_pages = extract_pdf_pages(pdf_bytes)
    assert len(pdf_pages) == 3

    mock_llm = _make_mock_llm_full_pipeline()

    result = run_pipeline(claim_id="CLAIM-001", pdf_pages=pdf_pages, llm=mock_llm)

    # Result type and claim_id
    assert isinstance(result, ClaimResult)
    assert result.claim_id == "CLAIM-001"

    # page_classification_map is present and covers all 3 pages
    assert result.page_classification_map is not None
    all_classified = [
        idx
        for pages in result.page_classification_map.values()
        for idx in pages
    ]
    assert sorted(all_classified) == [0, 1, 2]

    # Identity section populated
    assert result.identity is not None
    assert result.identity.patient_name == "Alice"

    # Discharge section populated
    assert result.discharge_summary is not None
    assert result.discharge_summary.diagnosis == "Flu"

    # Bill section populated
    assert result.itemized_bill is not None
    assert len(result.itemized_bill.line_items) == 1
    assert result.itemized_bill.line_items[0].description == "X-Ray"
    assert result.itemized_bill.total_amount == 150.0


# ---------------------------------------------------------------------------
# Task 8.2 — PDF with no identity pages produces null identity section
# ---------------------------------------------------------------------------

def test_pipeline_no_identity_pages_produces_null_identity():
    """
    Integration test: when no pages are classified as identity_document,
    the identity section of ClaimResult must be None.

    Validates: Requirements 2.5, 3, 6.3
    """
    pdf_bytes = _make_pdf_bytes(2)
    pdf_pages = extract_pdf_pages(pdf_bytes)
    assert len(pdf_pages) == 2

    mock_llm = _make_mock_llm_no_identity()

    result = run_pipeline(claim_id="CLAIM-002", pdf_pages=pdf_pages, llm=mock_llm)

    assert isinstance(result, ClaimResult)
    assert result.claim_id == "CLAIM-002"

    # No identity pages → identity section must be None
    assert result.identity is None

    # Other sections should be populated
    assert result.discharge_summary is not None
    assert result.discharge_summary.diagnosis == "Cold"

    assert result.itemized_bill is not None
    assert result.itemized_bill.total_amount == 80.0


# ---------------------------------------------------------------------------
# Task 8.3 — PDF with no itemized bill pages produces null bill section
# ---------------------------------------------------------------------------

def test_pipeline_no_bill_pages_produces_null_bill_section():
    """
    Integration test: when no pages are classified as itemized_bill,
    the itemized_bill section of ClaimResult must be None (agent not invoked).

    Validates: Requirements 2.5, 5, 6.3
    """
    pdf_bytes = _make_pdf_bytes(2)
    pdf_pages = extract_pdf_pages(pdf_bytes)
    assert len(pdf_pages) == 2

    mock_llm = _make_mock_llm_no_bill()

    result = run_pipeline(claim_id="CLAIM-003", pdf_pages=pdf_pages, llm=mock_llm)

    assert isinstance(result, ClaimResult)
    assert result.claim_id == "CLAIM-003"

    # No bill pages → bill section must be None
    assert result.itemized_bill is None

    # Other sections should be populated
    assert result.identity is not None
    assert result.identity.patient_name == "Bob"

    assert result.discharge_summary is not None
    assert result.discharge_summary.diagnosis == "Sprain"
