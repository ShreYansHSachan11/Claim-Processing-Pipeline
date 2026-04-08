"""Segregator node — classifies PDF pages using vision LLM in batches of 5."""

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

MAX_IMAGES_PER_CALL = 5


class PageClassification(BaseModel):
    document_type: DocumentType


class AllPagesClassification(BaseModel):
    classifications: list[PageClassification]


def _classify_batch(llm: BaseChatModel, b64_images: list[str], start_idx: int) -> list[tuple[int, str]]:
    """Classify a batch of pages (max 5) in a single LLM call."""
    n = len(b64_images)
    prompt = (
        f"You are a document classifier for insurance claims. "
        f"I am sending you {n} page image(s) (pages {start_idx} to {start_idx + n - 1}). "
        f"Classify EACH page into exactly one of these document types: "
        + ", ".join(DOCUMENT_TYPES)
        + f". Return a JSON with a 'classifications' array containing exactly {n} objects "
        f"in order, each with a 'document_type' field."
    )

    content = []
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})

    structured_llm = llm.with_structured_output(AllPagesClassification)
    result: AllPagesClassification = structured_llm.invoke([HumanMessage(content=content)])
    classifications = result.classifications

    # Pad with "other" if fewer results returned than pages sent
    while len(classifications) < n:
        classifications.append(PageClassification(document_type="other"))

    return [
        (start_idx + i, cls.document_type if cls.document_type in DOCUMENT_TYPES else "other")
        for i, cls in enumerate(classifications[:n])
    ]


def segregator_node(state: PipelineState, llm: BaseChatModel) -> dict:
    """Classify all PDF pages into document types, batching up to 5 images per call."""
    pdf_pages = state["pdf_pages"]
    n = len(pdf_pages)

    page_classification_map: dict[str, list[int]] = {doc_type: [] for doc_type in DOCUMENT_TYPES}

    if n == 0:
        return {"page_classification_map": page_classification_map}

    # Process in batches of MAX_IMAGES_PER_CALL
    for batch_start in range(0, n, MAX_IMAGES_PER_CALL):
        batch = pdf_pages[batch_start:batch_start + MAX_IMAGES_PER_CALL]
        try:
            results = _classify_batch(llm, batch, batch_start)
            for idx, doc_type in results:
                page_classification_map[doc_type].append(idx)
                print(f"[segregator] page {idx}: classified as '{doc_type}'")
        except Exception as e:
            print(f"[segregator] batch {batch_start}-{batch_start+len(batch)-1}: EXCEPTION — {type(e).__name__}: {e}")
            for i in range(len(batch)):
                page_classification_map["other"].append(batch_start + i)

    return {"page_classification_map": page_classification_map}
