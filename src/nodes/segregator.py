"""Segregator node — classifies all PDF pages in a single GPT-4o/Gemini vision call."""

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


class PageClassification(BaseModel):
    document_type: DocumentType


class AllPagesClassification(BaseModel):
    """Structured output: one classification per page, in order."""
    classifications: list[PageClassification]


def segregator_node(state: PipelineState, llm: BaseChatModel) -> dict:
    """Classify all PDF pages in a single LLM vision call."""
    pdf_pages = state["pdf_pages"]
    n = len(pdf_pages)

    page_classification_map: dict[str, list[int]] = {doc_type: [] for doc_type in DOCUMENT_TYPES}

    if n == 0:
        return {"page_classification_map": page_classification_map}

    prompt = (
        f"You are a document classifier for insurance claims. "
        f"I am sending you {n} page images from a single PDF (pages 0 to {n-1}). "
        f"Classify EACH page into exactly one of these document types: "
        + ", ".join(DOCUMENT_TYPES)
        + f". Return a JSON with a 'classifications' array containing exactly {n} objects, "
        f"one per page in order, each with a 'document_type' field."
    )

    # Build a single message with all page images + the prompt
    content = []
    for b64 in pdf_pages:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})

    try:
        structured_llm = llm.with_structured_output(AllPagesClassification)
        result: AllPagesClassification = structured_llm.invoke([HumanMessage(content=content)])
        classifications = result.classifications

        # Pad with "other" if LLM returned fewer classifications than pages
        while len(classifications) < n:
            classifications.append(PageClassification(document_type="other"))

        for idx, cls in enumerate(classifications[:n]):
            doc_type = cls.document_type if cls.document_type in DOCUMENT_TYPES else "other"
            page_classification_map[doc_type].append(idx)
            print(f"[segregator] page {idx}: classified as '{doc_type}'")

    except Exception as e:
        print(f"[segregator] EXCEPTION — {type(e).__name__}: {e}")
        # Fallback: all pages go to "other"
        for idx in range(n):
            page_classification_map["other"].append(idx)

    return {"page_classification_map": page_classification_map}
