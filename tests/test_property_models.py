# Feature: claim-processing-pipeline, Property 8: claim result round-trip serialization

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.models import (
    BillResult,
    ClaimResult,
    DischargeResult,
    IdentityResult,
    LineItem,
)

DOC_TYPES = [
    "claim_form",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

line_item_strategy = st.builds(
    LineItem,
    description=st.text(min_size=1),
    cost=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
)

identity_strategy = st.builds(
    IdentityResult,
    patient_name=st.one_of(st.none(), st.text()),
    date_of_birth=st.one_of(st.none(), st.text()),
    id_numbers=st.one_of(st.none(), st.lists(st.text())),
    policy_details=st.one_of(st.none(), st.text()),
)

discharge_strategy = st.builds(
    DischargeResult,
    diagnosis=st.one_of(st.none(), st.text()),
    admission_date=st.one_of(st.none(), st.text()),
    discharge_date=st.one_of(st.none(), st.text()),
    attending_physician=st.one_of(st.none(), st.text()),
)


@st.composite
def bill_result_strategy(draw):
    items = draw(st.lists(line_item_strategy, max_size=20))
    total = sum(item.cost for item in items)
    return BillResult(line_items=items, total_amount=total)


claim_result_strategy = st.builds(
    ClaimResult,
    claim_id=st.text(min_size=1),
    page_classification_map=st.dictionaries(
        keys=st.sampled_from(DOC_TYPES),
        values=st.lists(st.integers(min_value=0, max_value=100)),
    ),
    identity=st.one_of(st.none(), identity_strategy),
    discharge_summary=st.one_of(st.none(), discharge_strategy),
    itemized_bill=st.one_of(st.none(), bill_result_strategy()),
)


# Validates: Requirements 1.2, 6.1
@given(result=claim_result_strategy)
@settings(max_examples=100)
def test_claim_result_roundtrip(result: ClaimResult):
    """Property 8: Any valid ClaimResult survives a JSON round-trip with no data loss."""
    json_str = result.model_dump_json()
    restored = ClaimResult.model_validate_json(json_str)
    assert restored == result
