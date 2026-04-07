# Feature: claim-processing-pipeline, Property 4: missing fields are null not absent
# Feature: claim-processing-pipeline, Property 5: bill total equals sum of line items
# Validates: Requirements 3, 4, 5

import math
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import BillResult, DischargeResult, IdentityResult, LineItem
from src.nodes.extraction_agents import (
    discharge_summary_agent_node,
    id_agent_node,
    itemized_bill_agent_node,
)

IDENTITY_FIELDS = list(IdentityResult.model_fields.keys())
DISCHARGE_FIELDS = list(DischargeResult.model_fields.keys())


def _make_state(page_texts: list[str], doc_type: str) -> dict:
    indices = list(range(len(page_texts)))
    return {
        "claim_id": "test",
        "pdf_pages": page_texts,
        "page_classification_map": {doc_type: indices},
        "identity_result": None,
        "discharge_result": None,
        "bill_result": None,
        "final_result": None,
    }


def _mock_llm_returning(value) -> MagicMock:
    mock_llm = MagicMock()
    structured = MagicMock()
    mock_llm.with_structured_output.return_value = structured
    structured.invoke.return_value = value
    return mock_llm


# ---------------------------------------------------------------------------
# Property 4: Missing fields are null, not absent
# ---------------------------------------------------------------------------

@given(page_batch=st.lists(st.text()))
@settings(max_examples=100)
def test_identity_fields_present_or_null(page_batch: list[str]) -> None:
    """Property 4: every declared IdentityResult field is present and null when missing."""
    all_none = IdentityResult()  # all fields default to None
    mock_llm = _mock_llm_returning(all_none)
    state = _make_state(page_batch, "identity_document")

    result = id_agent_node(state, mock_llm)  # type: ignore[arg-type]
    identity = result["identity_result"]
    result_dict = identity.model_dump()

    for field in IDENTITY_FIELDS:
        assert field in result_dict, f"Field '{field}' is absent from IdentityResult"
        assert result_dict[field] is None, (
            f"Field '{field}' should be None when missing, got {result_dict[field]!r}"
        )


@given(page_batch=st.lists(st.text()))
@settings(max_examples=100)
def test_discharge_fields_present_or_null(page_batch: list[str]) -> None:
    """Property 4: every declared DischargeResult field is present and null when missing."""
    all_none = DischargeResult()  # all fields default to None
    mock_llm = _mock_llm_returning(all_none)
    state = _make_state(page_batch, "discharge_summary")

    result = discharge_summary_agent_node(state, mock_llm)  # type: ignore[arg-type]
    discharge = result["discharge_result"]
    result_dict = discharge.model_dump()

    for field in DISCHARGE_FIELDS:
        assert field in result_dict, f"Field '{field}' is absent from DischargeResult"
        assert result_dict[field] is None, (
            f"Field '{field}' should be None when missing, got {result_dict[field]!r}"
        )


# ---------------------------------------------------------------------------
# Property 5: Bill total equals sum of line items
# ---------------------------------------------------------------------------

line_item_strategy = st.builds(
    LineItem,
    description=st.text(min_size=1),
    cost=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)


@given(line_items=st.lists(line_item_strategy))
@settings(max_examples=100)
def test_bill_total_equals_sum(line_items: list[LineItem]) -> None:
    """Property 5: bill total_amount equals arithmetic sum of all line item costs."""
    mock_llm = _mock_llm_returning(BillResult(line_items=line_items, total_amount=0.0))
    state = _make_state(["some bill page"], "itemized_bill")

    result = itemized_bill_agent_node(state, mock_llm)  # type: ignore[arg-type]
    bill = result["bill_result"]

    expected_total = sum(item.cost for item in line_items)
    assert math.isclose(bill.total_amount, expected_total, rel_tol=1e-9, abs_tol=1e-9), (
        f"total_amount {bill.total_amount} != sum of costs {expected_total}"
    )


def test_bill_empty_list_gives_zero_total() -> None:
    """Property 5 edge case: empty line items → total is 0.0."""
    mock_llm = _mock_llm_returning(BillResult(line_items=[], total_amount=999.0))
    state = _make_state([], "itemized_bill")

    result = itemized_bill_agent_node(state, mock_llm)  # type: ignore[arg-type]
    bill = result["bill_result"]

    assert bill.line_items == []
    assert bill.total_amount == 0.0
