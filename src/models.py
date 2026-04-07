from typing import Optional, Literal
from pydantic import BaseModel

DocumentType = Literal[
    "claim_form", "cheque_or_bank_details", "identity_document",
    "itemized_bill", "discharge_summary", "prescription",
    "investigation_report", "cash_receipt", "other"
]


class IdentityResult(BaseModel):
    patient_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    id_numbers: Optional[list[str]] = None
    policy_details: Optional[str] = None


class DischargeResult(BaseModel):
    diagnosis: Optional[str] = None
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    attending_physician: Optional[str] = None


class LineItem(BaseModel):
    description: str
    cost: float


class BillResult(BaseModel):
    line_items: list[LineItem] = []
    total_amount: float = 0.0


class ClaimResult(BaseModel):
    claim_id: str
    page_classification_map: dict[str, list[int]]
    identity: Optional[IdentityResult] = None
    discharge_summary: Optional[DischargeResult] = None
    itemized_bill: Optional[BillResult] = None
