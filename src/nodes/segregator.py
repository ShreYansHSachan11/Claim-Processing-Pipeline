"""Segregator node — classifies each PDF page image using GPT-4o vision."""

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.state import PipelineState
from src.models import DocumentType

DOCUMENT_TYPES = [
    "claim_form", "cheque_or_bank_details", "identity_document",
    "itemized_bill", "discharge_summary", "prescription",
    "investigation_report", "cash_receipt", "other"
]

CLASSIFICATION_PROMPT = (
    "You are a document classifier for insurance claims. "
    "Look at this page image and classify it into exactly one of these document types: "
    + ", ".join(DOCUMENT_TYPES)
    + ". Respond with only the document_type field."
)


class PageClassification(BaseModel):
    document_type: DocumentType


def _vision_message(b64_image: str, prompt: str) -> HumanMessage:
    """Build a LangChain HumanMessage with an image + text for vision models."""
    return HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
        {"type": "text", "text": prompt},
    ])


def segregator_node(state: PipelineState, llm: BaseChatModel) -> dict:
    """Classify each PDF page image into a DocumentType using GPT-4o vision."""
    pdf_pages = state["pdf_pages"]
    structured_llm = llm.with_structured_output(PageClassification)

    page_classification_map: dict[str, list[int]] = {doc_type: [] for doc_type in DOCUMENT_TYPES}

    for idx, b64_image in enumerate(pdf_pages):
        try:
            msg = _vision_message(b64_image, CLASSIFICATION_PROMPT)
            result: PageClassification = structured_llm.invoke([msg])
            doc_type = result.document_type
            if doc_type not in DOCUMENT_TYPES:
                doc_type = "other"
            print(f"[segregator] page {idx}: classified as '{doc_type}'")
        except Exception as e:
            print(f"[segregator] page {idx}: EXCEPTION — {type(e).__name__}: {e}")
            doc_type = "other"

        page_classification_map[doc_type].append(idx)

    return {"page_classification_map": page_classification_map}
