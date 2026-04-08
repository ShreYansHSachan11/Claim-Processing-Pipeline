"""Segregator node — classifies PDF pages using vision LLM in batches of 5."""

import json
import re
from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage

from src.state import PipelineState
from src.models import DocumentType

DOCUMENT_TYPES = [
    "claim_form", "cheque_or_bank_details", "identity_document",
    "itemized_bill", "discharge_summary", "prescription",
    "investigation_report", "cash_receipt", "other"
]

MAX_IMAGES_PER_CALL = 5


def _extract_json_from_text(text: str) -> list | None:
    """Try to extract a JSON array from any text."""
    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "classifications" in parsed:
            return parsed["classifications"]
    except Exception:
        pass

    # Find JSON array in text
    match = re.search(r'\[\s*\{[\s\S]*?\}\s*\]', text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

    return None


def _classify_batch(llm: BaseChatModel, b64_images: list[str], start_idx: int) -> list[tuple[int, str]]:
    """Classify a batch of pages in a single LLM call, with robust JSON parsing."""
    n = len(b64_images)
    prompt = (
        f"You are a document classifier for insurance claims. "
        f"Classify each of the {n} page image(s) I'm sending into exactly one of: "
        + ", ".join(DOCUMENT_TYPES)
        + f". Respond with ONLY a JSON array of {n} objects like: "
        + '[{"document_type": "claim_form"}, {"document_type": "other"}]'
        + ". No explanation, just the JSON array."
    )

    content = []
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})

    def _parse_results(raw: list) -> list[tuple[int, str]]:
        while len(raw) < n:
            raw.append({"document_type": "other"})
        results = []
        for i, item in enumerate(raw[:n]):
            if isinstance(item, dict):
                doc_type = item.get("document_type", "other")
            else:
                doc_type = str(item)
            if doc_type not in DOCUMENT_TYPES:
                doc_type = "other"
            results.append((start_idx + i, doc_type))
        return results

    # Use plain invoke (no structured output) to avoid Groq tool_use_failed errors
    try:
        response = llm.invoke([HumanMessage(content=content)])
        text = response.content if hasattr(response, "content") else str(response)
        raw = _extract_json_from_text(text)
        if raw is not None:
            return _parse_results(raw)
    except Exception as e:
        # Try to extract from error's failed_generation
        err_str = str(e)
        raw = _extract_json_from_text(err_str)
        if raw is not None:
            return _parse_results(raw)
        raise

    # Fallback: all pages in this batch go to "other"
    return [(start_idx + i, "other") for i in range(n)]


def segregator_node(state: PipelineState, llm: BaseChatModel) -> dict:
    """Classify all PDF pages into document types, batching up to 5 images per call."""
    pdf_pages = state["pdf_pages"]
    n = len(pdf_pages)

    page_classification_map: dict[str, list[int]] = {doc_type: [] for doc_type in DOCUMENT_TYPES}

    if n == 0:
        return {"page_classification_map": page_classification_map}

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
