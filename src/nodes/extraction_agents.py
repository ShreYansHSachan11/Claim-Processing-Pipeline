"""Extraction agent nodes — use GPT-4o vision to extract structured data from page images."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from src.state import PipelineState
from src.models import IdentityResult, DischargeResult, BillResult

IDENTITY_PROMPT = (
    "You are an insurance claim processor. "
    "Extract identity information from these document page images. "
    "Return all fields you can find. Set fields to null if not present."
)

DISCHARGE_PROMPT = (
    "You are an insurance claim processor. "
    "Extract discharge summary information from these document page images. "
    "Return all fields you can find. Set fields to null if not present."
)

BILL_PROMPT = (
    "You are an insurance claim processor. "
    "Extract all itemized bill line items from these document page images. "
    "Return every line item with its description and cost. "
    "Return an empty list if no line items are found."
)


def _vision_messages(b64_images: list[str], prompt: str) -> list[HumanMessage]:
    """Build a single HumanMessage containing all page images + the prompt."""
    content = []
    for b64 in b64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    content.append({"type": "text", "text": prompt})
    return [HumanMessage(content=content)]


def _get_page_images(state: PipelineState, doc_type: str) -> list[str]:
    """Return the base64 page images for the given document type."""
    indices = state["page_classification_map"].get(doc_type, [])
    pdf_pages = state["pdf_pages"]
    return [pdf_pages[i] for i in indices if i < len(pdf_pages)]


def id_agent_node(state: PipelineState, llm: BaseChatModel) -> dict:
    """Extract identity fields from identity_document page images."""
    images = _get_page_images(state, "identity_document")
    structured_llm = llm.with_structured_output(IdentityResult)
    try:
        msgs = _vision_messages(images, IDENTITY_PROMPT) if images else [
            HumanMessage(content=IDENTITY_PROMPT + "\n\nNo pages provided.")
        ]
        result: IdentityResult = structured_llm.invoke(msgs)
    except Exception:
        result = IdentityResult()
    return {"identity_result": result}


def discharge_summary_agent_node(state: PipelineState, llm: BaseChatModel) -> dict:
    """Extract clinical fields from discharge_summary page images."""
    images = _get_page_images(state, "discharge_summary")
    structured_llm = llm.with_structured_output(DischargeResult)
    try:
        msgs = _vision_messages(images, DISCHARGE_PROMPT) if images else [
            HumanMessage(content=DISCHARGE_PROMPT + "\n\nNo pages provided.")
        ]
        result: DischargeResult = structured_llm.invoke(msgs)
    except Exception:
        result = DischargeResult()
    return {"discharge_result": result}


def itemized_bill_agent_node(state: PipelineState, llm: BaseChatModel) -> dict:
    """Extract line items from itemized_bill page images and compute total."""
    images = _get_page_images(state, "itemized_bill")
    structured_llm = llm.with_structured_output(BillResult)
    try:
        msgs = _vision_messages(images, BILL_PROMPT) if images else [
            HumanMessage(content=BILL_PROMPT + "\n\nNo pages provided.")
        ]
        raw: BillResult = structured_llm.invoke(msgs)
        line_items = raw.line_items if raw.line_items else []
    except Exception:
        line_items = []

    # Always recompute total — never trust LLM's total_amount
    total = sum(item.cost for item in line_items)
    return {"bill_result": BillResult(line_items=line_items, total_amount=total)}
