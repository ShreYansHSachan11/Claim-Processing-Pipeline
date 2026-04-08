"""Extraction agent nodes — use vision LLM to extract structured data from page images."""

import json
import re
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.state import PipelineState
from src.models import IdentityResult, DischargeResult, BillResult, LineItem

IDENTITY_PROMPT = """You are an insurance claim processor. Extract identity information from these document page images.
Return ONLY a JSON object with these exact fields:
{"patient_name": "...", "date_of_birth": "...", "id_numbers": ["..."], "policy_details": "..."}
Use null for any field not found. No explanation, just the JSON object."""

DISCHARGE_PROMPT = """You are an insurance claim processor. Extract discharge summary information from these document page images.
Return ONLY a JSON object with these exact fields:
{"diagnosis": "...", "admission_date": "...", "discharge_date": "...", "attending_physician": "..."}
Use null for any field not found. No explanation, just the JSON object."""

BILL_PROMPT = """You are an insurance claim processor. Extract all itemized bill line items from these document page images.
Return ONLY a JSON object with this exact structure:
{"line_items": [{"description": "...", "cost": 0.0}], "total_amount": 0.0}
Return empty list if no line items found. No explanation, just the JSON object."""


def _extract_json(text: str) -> dict | None:
    """Extract first JSON object from text."""
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Find JSON object in text
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def _build_messages(b64_images: list[str], prompt: str) -> list[HumanMessage]:
    content = []
    for b64 in b64_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    return [HumanMessage(content=content)]


def _get_page_images(state: PipelineState, doc_type: str) -> list[str]:
    indices = state["page_classification_map"].get(doc_type, [])
    return [state["pdf_pages"][i] for i in indices if i < len(state["pdf_pages"])]


def id_agent_node(state: PipelineState, llm: BaseChatModel) -> dict:
    images = _get_page_images(state, "identity_document")
    try:
        if images:
            response = llm.invoke(_build_messages(images, IDENTITY_PROMPT))
            data = _extract_json(response.content if hasattr(response, "content") else str(response))
            if data:
                result = IdentityResult(
                    patient_name=data.get("patient_name"),
                    date_of_birth=data.get("date_of_birth"),
                    id_numbers=data.get("id_numbers"),
                    policy_details=data.get("policy_details"),
                )
                return {"identity_result": result}
    except Exception as e:
        print(f"[id_agent] EXCEPTION: {e}")
    return {"identity_result": IdentityResult()}


def discharge_summary_agent_node(state: PipelineState, llm: BaseChatModel) -> dict:
    images = _get_page_images(state, "discharge_summary")
    try:
        if images:
            response = llm.invoke(_build_messages(images, DISCHARGE_PROMPT))
            data = _extract_json(response.content if hasattr(response, "content") else str(response))
            if data:
                result = DischargeResult(
                    diagnosis=data.get("diagnosis"),
                    admission_date=data.get("admission_date"),
                    discharge_date=data.get("discharge_date"),
                    attending_physician=data.get("attending_physician"),
                )
                return {"discharge_result": result}
    except Exception as e:
        print(f"[discharge_agent] EXCEPTION: {e}")
    return {"discharge_result": DischargeResult()}


def itemized_bill_agent_node(state: PipelineState, llm: BaseChatModel) -> dict:
    images = _get_page_images(state, "itemized_bill")
    try:
        if images:
            response = llm.invoke(_build_messages(images, BILL_PROMPT))
            data = _extract_json(response.content if hasattr(response, "content") else str(response))
            if data:
                raw_items = data.get("line_items", [])
                line_items = [
                    LineItem(description=item.get("description", ""), cost=float(item.get("cost", 0)))
                    for item in raw_items if isinstance(item, dict)
                ]
                total = sum(item.cost for item in line_items)
                return {"bill_result": BillResult(line_items=line_items, total_amount=total)}
    except Exception as e:
        print(f"[bill_agent] EXCEPTION: {e}")
    return {"bill_result": BillResult(line_items=[], total_amount=0.0)}
