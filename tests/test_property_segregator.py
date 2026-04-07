# Feature: claim-processing-pipeline, Property 1: every page classified exactly once
# Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5

from unittest.mock import MagicMock
from hypothesis import given, settings
from hypothesis import strategies as st

from src.nodes.segregator import segregator_node, DOCUMENT_TYPES, PageClassification


def make_mock_llm(pdf_pages: list[str]) -> MagicMock:
    """Build a mock LLM that returns a random (but valid) DocumentType for each page."""
    mock_llm = MagicMock()
    structured = MagicMock()
    mock_llm.with_structured_output.return_value = structured

    # Generate one classification per page using a cycling strategy
    classifications = [
        PageClassification(document_type=DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)])
        for i in range(len(pdf_pages))
    ]
    structured.invoke.side_effect = classifications
    return mock_llm


@given(pdf_pages=st.lists(st.text(), min_size=1, max_size=20))
@settings(max_examples=100)
def test_segregator_covers_all_pages(pdf_pages: list[str]) -> None:
    """Property 1: every page index 0..N-1 appears exactly once across all buckets."""
    mock_llm = make_mock_llm(pdf_pages)
    state = {
        "claim_id": "test-claim",
        "pdf_pages": pdf_pages,
        "page_classification_map": {},
        "identity_result": None,
        "discharge_result": None,
        "bill_result": None,
        "final_result": None,
    }

    result = segregator_node(state, mock_llm)  # type: ignore[arg-type]
    classification_map = result["page_classification_map"]

    # Collect all page indices across all buckets
    all_indices = []
    for indices in classification_map.values():
        all_indices.extend(indices)

    expected = set(range(len(pdf_pages)))

    # Every page index must appear exactly once
    assert len(all_indices) == len(pdf_pages), (
        f"Expected {len(pdf_pages)} total indices, got {len(all_indices)}"
    )
    assert set(all_indices) == expected, (
        f"Expected indices {expected}, got {set(all_indices)}"
    )
    # No duplicates
    assert len(all_indices) == len(set(all_indices)), (
        f"Duplicate page indices found: {all_indices}"
    )


@given(pdf_pages=st.lists(st.text(), min_size=1, max_size=10))
@settings(max_examples=50)
def test_segregator_defaults_to_other_on_failure(pdf_pages: list[str]) -> None:
    """Property 1 + Req 2.3: pages that fail classification land in 'other'."""
    mock_llm = MagicMock()
    structured = MagicMock()
    mock_llm.with_structured_output.return_value = structured
    # Always raise an exception to simulate LLM failure
    structured.invoke.side_effect = Exception("LLM unavailable")

    state = {
        "claim_id": "test-claim",
        "pdf_pages": pdf_pages,
        "page_classification_map": {},
        "identity_result": None,
        "discharge_result": None,
        "bill_result": None,
        "final_result": None,
    }

    result = segregator_node(state, mock_llm)  # type: ignore[arg-type]
    classification_map = result["page_classification_map"]

    # All pages must end up in "other"
    assert classification_map["other"] == list(range(len(pdf_pages)))

    # Total indices still covers all pages exactly once
    all_indices = [i for indices in classification_map.values() for i in indices]
    assert sorted(all_indices) == list(range(len(pdf_pages)))
