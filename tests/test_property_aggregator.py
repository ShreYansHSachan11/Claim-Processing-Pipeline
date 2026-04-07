# Feature: claim-processing-pipeline, Property 7: aggregator output always contains all required sections
# Validates: Requirements 6.1, 6.2, 6.3

from typing import Optional

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import BillResult, ClaimResult, DischargeResult, IdentityResult, LineItem
from src.nodes.aggregator import aggregator_node

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

identity_strategy = st.one_of(
    st.none(),
    st.builds(
        IdentityResult,
        patient_name=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
        date_of_birth=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        id_numbers=st.one_of(st.none(), st.lists(st.text(min_size=1, max_size=20))),
        policy_details=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    ),
)

discharge_strategy = st.one_of(
    st.none(),
    st.builds(
        DischargeResult,
        diagnosis=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
        admission_date=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        discharge_date=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        attending_physician=st.one_of(st.none(), st.text(min_size=1, max_size=100)),
    ),
)

line_item_strategy = st.builds(
    LineItem,
    description=st.text(min_size=1, max_size=50),
    cost=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
)

bill_strategy = st.one_of(
    st.none(),
    st.builds(
        BillResult,
        line_items=st.lists(line_item_strategy),
        total_amount=st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False),
    ),
)

page_classification_map_strategy = st.dictionaries(
    keys=st.text(min_size=1, max_size=30),
    values=st.lists(st.integers(min_value=0, max_value=100)),
    min_size=0,
    max_size=5,
)


def _make_state(
    claim_id: str,
    page_classification_map: dict,
    identity_result: Optional[IdentityResult],
    discharge_result: Optional[DischargeResult],
    bill_result: Optional[BillResult],
) -> dict:
    return {
        "claim_id": claim_id,
        "pdf_pages": [],
        "page_classification_map": page_classification_map,
        "identity_result": identity_result,
        "discharge_result": discharge_result,
        "bill_result": bill_result,
        "final_result": None,
    }


# ---------------------------------------------------------------------------
# Property 7: Aggregator output always contains all required sections
# ---------------------------------------------------------------------------

@given(
    claim_id=st.text(min_size=1, max_size=50),
    page_classification_map=page_classification_map_strategy,
    identity_result=identity_strategy,
    discharge_result=discharge_strategy,
    bill_result=bill_strategy,
)
@settings(max_examples=100)
def test_aggregator_output_contains_all_required_sections(
    claim_id: str,
    page_classification_map: dict,
    identity_result: Optional[IdentityResult],
    discharge_result: Optional[DischargeResult],
    bill_result: Optional[BillResult],
) -> None:
    """Property 7: ClaimResult always has claim_id, page_classification_map, and all three agent sections."""
    state = _make_state(claim_id, page_classification_map, identity_result, discharge_result, bill_result)

    output = aggregator_node(state)
    result = output["final_result"]

    # Must be a ClaimResult
    assert isinstance(result, ClaimResult)

    # Required fields must be present (6.1, 6.2)
    assert result.claim_id == claim_id
    assert result.page_classification_map == page_classification_map

    # All three agent sections must be present as keys (6.1, 6.3)
    result_dict = result.model_dump()
    assert "identity" in result_dict
    assert "discharge_summary" in result_dict
    assert "itemized_bill" in result_dict


@given(
    claim_id=st.text(min_size=1, max_size=50),
    page_classification_map=page_classification_map_strategy,
    identity_result=identity_strategy,
    discharge_result=discharge_strategy,
    bill_result=bill_strategy,
)
@settings(max_examples=100)
def test_aggregator_sections_match_state_values(
    claim_id: str,
    page_classification_map: dict,
    identity_result: Optional[IdentityResult],
    discharge_result: Optional[DischargeResult],
    bill_result: Optional[BillResult],
) -> None:
    """Property 7: sections are None when state value is None, populated when state value is provided."""
    state = _make_state(claim_id, page_classification_map, identity_result, discharge_result, bill_result)

    output = aggregator_node(state)
    result = output["final_result"]

    # Sections are None iff the corresponding state value was None (6.3)
    if identity_result is None:
        assert result.identity is None
    else:
        assert result.identity == identity_result

    if discharge_result is None:
        assert result.discharge_summary is None
    else:
        assert result.discharge_summary == discharge_result

    if bill_result is None:
        assert result.itemized_bill is None
    else:
        assert result.itemized_bill == bill_result


# ---------------------------------------------------------------------------
# Unit tests for the aggregator
# ---------------------------------------------------------------------------

def test_aggregator_all_agents_ran() -> None:
    """When all three agents ran, all sections are populated in the result."""
    identity = IdentityResult(patient_name="Alice", date_of_birth="1990-01-01")
    discharge = DischargeResult(diagnosis="Flu", admission_date="2024-01-01", discharge_date="2024-01-05")
    bill = BillResult(line_items=[LineItem(description="X-Ray", cost=150.0)], total_amount=150.0)
    pcm = {"identity_document": [0], "discharge_summary": [1], "itemized_bill": [2]}

    state = _make_state("CLAIM-001", pcm, identity, discharge, bill)
    output = aggregator_node(state)
    result = output["final_result"]

    assert isinstance(result, ClaimResult)
    assert result.claim_id == "CLAIM-001"
    assert result.page_classification_map == pcm
    assert result.identity == identity
    assert result.discharge_summary == discharge
    assert result.itemized_bill == bill


def test_aggregator_no_agents_ran() -> None:
    """When no agents ran (all None in state), all sections are None in the result."""
    pcm = {"identity_document": [], "discharge_summary": [], "itemized_bill": []}

    state = _make_state("CLAIM-002", pcm, None, None, None)
    output = aggregator_node(state)
    result = output["final_result"]

    assert isinstance(result, ClaimResult)
    assert result.claim_id == "CLAIM-002"
    assert result.page_classification_map == pcm
    assert result.identity is None
    assert result.discharge_summary is None
    assert result.itemized_bill is None
