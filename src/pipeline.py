from functools import partial
from typing import Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.language_models import BaseChatModel

from src.state import PipelineState
from src.models import ClaimResult
from src.nodes.segregator import segregator_node
from src.nodes.extraction_agents import (
    id_agent_node,
    discharge_summary_agent_node,
    itemized_bill_agent_node,
)
from src.nodes.aggregator import aggregator_node


def _has_identity_pages(state: PipelineState) -> str:
    pages = state.get("page_classification_map", {}).get("identity_document", [])
    return "id_agent" if pages else "aggregator"


def _has_discharge_pages(state: PipelineState) -> str:
    pages = state.get("page_classification_map", {}).get("discharge_summary", [])
    return "discharge_agent" if pages else "aggregator"


def _has_bill_pages(state: PipelineState) -> str:
    pages = state.get("page_classification_map", {}).get("itemized_bill", [])
    return "bill_agent" if pages else "aggregator"


def build_graph(llm: BaseChatModel) -> StateGraph:
    """Build and compile the LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Register nodes (bind LLM via partial where needed)
    graph.add_node("segregator", partial(segregator_node, llm=llm))
    graph.add_node("id_agent", partial(id_agent_node, llm=llm))
    graph.add_node("discharge_agent", partial(discharge_summary_agent_node, llm=llm))
    graph.add_node("bill_agent", partial(itemized_bill_agent_node, llm=llm))
    graph.add_node("aggregator", aggregator_node)

    # START → segregator
    graph.add_edge(START, "segregator")

    # Fan-out: segregator → agents (conditional — skip if no pages for that type)
    graph.add_conditional_edges(
        "segregator",
        _has_identity_pages,
        {"id_agent": "id_agent", "aggregator": "aggregator"},
    )
    graph.add_conditional_edges(
        "segregator",
        _has_discharge_pages,
        {"discharge_agent": "discharge_agent", "aggregator": "aggregator"},
    )
    graph.add_conditional_edges(
        "segregator",
        _has_bill_pages,
        {"bill_agent": "bill_agent", "aggregator": "aggregator"},
    )

    # Fan-in: agents → aggregator
    graph.add_edge("id_agent", "aggregator")
    graph.add_edge("discharge_agent", "aggregator")
    graph.add_edge("bill_agent", "aggregator")

    # aggregator → END
    graph.add_edge("aggregator", END)

    return graph.compile()


def run_pipeline(
    claim_id: str,
    pdf_pages: list[str],
    llm: Optional[BaseChatModel] = None,
) -> ClaimResult:
    """Run the claim processing pipeline and return the aggregated ClaimResult."""
    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    compiled = build_graph(llm)

    initial_state: PipelineState = {
        "claim_id": claim_id,
        "pdf_pages": pdf_pages,
        "page_classification_map": {},
        "identity_result": None,
        "discharge_result": None,
        "bill_result": None,
        "final_result": None,
    }

    final_state = compiled.invoke(initial_state)
    return final_state["final_result"]
