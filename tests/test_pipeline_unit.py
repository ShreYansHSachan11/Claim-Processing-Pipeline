"""Unit tests for the LangGraph pipeline assembly (tasks 5.1–5.6)."""
from unittest.mock import MagicMock, patch, call

import pytest

from src.models import IdentityResult, DischargeResult, BillResult, ClaimResult
from src.state import PipelineState


def _make_mock_llm():
    """Return a mock LLM that won't be called in real tests."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Task 5.6 — agents are NOT invoked when their doc type has no pages
# ---------------------------------------------------------------------------

def test_id_agent_not_invoked_when_no_identity_pages():
    """
    When page_classification_map has no identity_document pages,
    id_agent_node must not be called.
    """
    # Arrange: segregator returns a map with identity_document empty
    mock_llm = _make_mock_llm()

    segregator_result = {
        "claim_form": [],
        "cheque_or_bank_details": [],
        "identity_document": [],          # <-- no pages
        "itemized_bill": [0],
        "discharge_summary": [1],
        "prescription": [],
        "investigation_report": [],
        "cash_receipt": [],
        "other": [],
    }

    discharge_result = DischargeResult(diagnosis="Flu")
    bill_result = BillResult(line_items=[], total_amount=0.0)

    with (
        patch("src.pipeline.segregator_node", return_value={"page_classification_map": segregator_result}) as mock_seg,
        patch("src.pipeline.id_agent_node", return_value={"identity_result": IdentityResult()}) as mock_id,
        patch("src.pipeline.discharge_summary_agent_node", return_value={"discharge_result": discharge_result}) as mock_dis,
        patch("src.pipeline.itemized_bill_agent_node", return_value={"bill_result": bill_result}) as mock_bill,
    ):
        from src.pipeline import run_pipeline

        result = run_pipeline(
            claim_id="CLAIM-001",
            pdf_pages=["page0 text", "page1 text"],
            llm=mock_llm,
        )

    # id_agent_node must NOT have been called
    mock_id.assert_not_called()

    # The other two agents should have been called
    mock_dis.assert_called_once()
    mock_bill.assert_called_once()

    # Result is a ClaimResult
    assert isinstance(result, ClaimResult)
    assert result.claim_id == "CLAIM-001"
    # identity section should be None (agent was skipped)
    assert result.identity is None


def test_discharge_agent_not_invoked_when_no_discharge_pages():
    """
    When page_classification_map has no discharge_summary pages,
    discharge_summary_agent_node must not be called.
    """
    mock_llm = _make_mock_llm()

    segregator_result = {
        "claim_form": [],
        "cheque_or_bank_details": [],
        "identity_document": [0],
        "itemized_bill": [1],
        "discharge_summary": [],          # <-- no pages
        "prescription": [],
        "investigation_report": [],
        "cash_receipt": [],
        "other": [],
    }

    identity_result = IdentityResult(patient_name="Alice")
    bill_result = BillResult(line_items=[], total_amount=0.0)

    with (
        patch("src.pipeline.segregator_node", return_value={"page_classification_map": segregator_result}),
        patch("src.pipeline.id_agent_node", return_value={"identity_result": identity_result}),
        patch("src.pipeline.discharge_summary_agent_node", return_value={"discharge_result": DischargeResult()}) as mock_dis,
        patch("src.pipeline.itemized_bill_agent_node", return_value={"bill_result": bill_result}),
    ):
        from src.pipeline import run_pipeline

        result = run_pipeline(
            claim_id="CLAIM-002",
            pdf_pages=["page0", "page1"],
            llm=mock_llm,
        )

    mock_dis.assert_not_called()
    assert result.discharge_summary is None


def test_bill_agent_not_invoked_when_no_bill_pages():
    """
    When page_classification_map has no itemized_bill pages,
    itemized_bill_agent_node must not be called.
    """
    mock_llm = _make_mock_llm()

    segregator_result = {
        "claim_form": [],
        "cheque_or_bank_details": [],
        "identity_document": [0],
        "itemized_bill": [],              # <-- no pages
        "discharge_summary": [1],
        "prescription": [],
        "investigation_report": [],
        "cash_receipt": [],
        "other": [],
    }

    identity_result = IdentityResult(patient_name="Bob")
    discharge_result = DischargeResult(diagnosis="Cold")

    with (
        patch("src.pipeline.segregator_node", return_value={"page_classification_map": segregator_result}),
        patch("src.pipeline.id_agent_node", return_value={"identity_result": identity_result}),
        patch("src.pipeline.discharge_summary_agent_node", return_value={"discharge_result": discharge_result}),
        patch("src.pipeline.itemized_bill_agent_node", return_value={"bill_result": BillResult()}) as mock_bill,
    ):
        from src.pipeline import run_pipeline

        result = run_pipeline(
            claim_id="CLAIM-003",
            pdf_pages=["page0", "page1"],
            llm=mock_llm,
        )

    mock_bill.assert_not_called()
    assert result.itemized_bill is None


def test_all_agents_invoked_when_all_doc_types_have_pages():
    """All three agents are called when every doc type has at least one page."""
    mock_llm = _make_mock_llm()

    segregator_result = {
        "claim_form": [],
        "cheque_or_bank_details": [],
        "identity_document": [0],
        "itemized_bill": [1],
        "discharge_summary": [2],
        "prescription": [],
        "investigation_report": [],
        "cash_receipt": [],
        "other": [],
    }

    with (
        patch("src.pipeline.segregator_node", return_value={"page_classification_map": segregator_result}),
        patch("src.pipeline.id_agent_node", return_value={"identity_result": IdentityResult()}) as mock_id,
        patch("src.pipeline.discharge_summary_agent_node", return_value={"discharge_result": DischargeResult()}) as mock_dis,
        patch("src.pipeline.itemized_bill_agent_node", return_value={"bill_result": BillResult()}) as mock_bill,
    ):
        from src.pipeline import run_pipeline

        result = run_pipeline(
            claim_id="CLAIM-004",
            pdf_pages=["p0", "p1", "p2"],
            llm=mock_llm,
        )

    mock_id.assert_called_once()
    mock_dis.assert_called_once()
    mock_bill.assert_called_once()
    assert isinstance(result, ClaimResult)
