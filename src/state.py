from typing import Optional, TypedDict

from src.models import BillResult, ClaimResult, DischargeResult, IdentityResult


class PipelineState(TypedDict):
    claim_id: str
    pdf_pages: list[str]  # base64-encoded PNG images, one per page
    page_classification_map: dict[str, list[int]]
    identity_result: Optional[IdentityResult]
    discharge_result: Optional[DischargeResult]
    bill_result: Optional[BillResult]
    final_result: Optional[ClaimResult]
