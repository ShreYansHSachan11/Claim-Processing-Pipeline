from src.state import PipelineState
from src.models import ClaimResult


def aggregator_node(state: PipelineState) -> dict:
    """Merge identity, discharge, and bill results into a final ClaimResult."""
    return {
        "final_result": ClaimResult(
            claim_id=state["claim_id"],
            page_classification_map=state["page_classification_map"],
            identity=state.get("identity_result"),
            discharge_summary=state.get("discharge_result"),
            itemized_bill=state.get("bill_result"),
        )
    }
