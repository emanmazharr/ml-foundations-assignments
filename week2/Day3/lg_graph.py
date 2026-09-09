"""
Week 2 / Day 3 — the LangGraph workflow itself.

Graph shape (see README.md / notebook for the ASCII + Mermaid diagrams):

    START -> search -> draft -> critique --(needs revision)--> draft   [loop]
                                     |
                                     --(good enough)--> human_approval
                                                              |
                                              (approved)------+------(rejected)
                                                   |                     |
                                               finalize                 END
                                                   |
                                                  END

Design notes:
- `critique` is a conditional-edge decision point: it can route back to
  `draft` (self-correction loop) or forward to `human_approval`.
- `retries` in State caps the loop so a persistently low-scoring draft
  can't cycle forever — once `max_retries` is hit, we force it forward to
  human_approval rather than looping again.
- `human_approval` calls `interrupt(...)`, which pauses the graph and
  requires a `Command(resume=...)` from the caller to continue — modeling
  a real "risky action" gate (here: sending a recommendation to a client)
  before anything is finalized.
- A checkpointer (`InMemorySaver`) is required for `interrupt` to work at
  all, and is what makes the graph's state persist across separate
  `.invoke()` calls keyed by the same `thread_id`.
"""

import json

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from lg_state import State, CritiqueResult
from lg_tools import lookup_product

QUALITY_THRESHOLD = 7  # scores >= this pass the critique without revision


def get_llm(model_name="gemini-3.6-flash", temperature=0):
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/app/apikey and set it before "
            "building the graph."
        )
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def _extract_text(response) -> str:
    """
    Safely pull the plain-text answer out of a chat model response,
    regardless of whether `.content` is a plain string (older/simpler
    models) or a list of content blocks (newer "thinking"-enabled models
    like Gemini 3.6, which interleave reasoning/thinking blocks alongside
    the actual text block). We only want the text block(s), never the
    thinking trace, and never a raw Python repr of the block list.
    """
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return str(content)


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def make_search_node():
    def search(state: State) -> dict:
        results = {}
        errors = []
        for p in (state["product_a"], state["product_b"]):
            try:
                results[p] = lookup_product(p)
            except KeyError as e:
                errors.append(str(e))
        search_data = json.dumps(results, indent=2)
        return {
            "search_data": search_data,
            "log": [f"[search] looked up {state['product_a']!r} and {state['product_b']!r}"
                    + (f" (errors: {errors})" if errors else "")],
        }

    return search


def make_draft_node(llm):
    def draft(state: State) -> dict:
        prior_feedback = (
            f"\n\nPrevious critique feedback to address: {state['critique_feedback']}"
            if state.get("critique_feedback")
            else ""
        )
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Catalog data:\n{state['search_data']}\n"
            f"{prior_feedback}\n\n"
            "Write a short, direct recommendation (3-4 sentences) answering "
            "the question, grounded ONLY in the catalog data above."
        )
        response = llm.invoke(prompt)
        pass_num = state.get("retries", 0) + 1
        return {
            "draft": _extract_text(response),
            "log": [f"[draft] pass {pass_num}: produced a new draft"],
        }

    return draft


def make_critique_node(llm):
    structured_llm = llm.with_structured_output(CritiqueResult)

    def critique(state: State) -> dict:
        prompt = (
            f"Question: {state['question']}\n\n"
            f"Draft recommendation:\n{state['draft']}\n\n"
            f"Catalog data it should be grounded in:\n{state['search_data']}\n\n"
            "Score this draft from 1-10 on accuracy (matches the catalog "
            "data exactly) and clarity. Set should_revise=true if score < "
            f"{QUALITY_THRESHOLD}."
        )
        result: CritiqueResult = structured_llm.invoke(prompt)
        return {
            "quality_score": result.score,
            "critique_feedback": result.feedback,
            "log": [f"[critique] score={result.score}, should_revise={result.should_revise}"],
        }

    return critique


def route_after_critique(state: State) -> str:
    """Conditional edge: loop back to draft, or move on to human approval."""
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "human_approval"
    if state["retries"] >= state["max_retries"]:
        # Give up looping — force it forward rather than cycling forever.
        return "human_approval"
    return "draft"


def increment_retries(state: State) -> dict:
    """A tiny bookkeeping node so `retries` only increments on an actual
    loop-back, not on the first pass through draft."""
    return {"retries": state["retries"] + 1, "log": [f"[loop] retry #{state['retries'] + 1}"]}


def human_approval(state: State) -> dict:
    """The interrupt point: pause before the 'risky action' (sending the
    recommendation to a client) and wait for a human decision."""
    decision = interrupt(
        {
            "action": "Send this recommendation to the client?",
            "draft": state["draft"],
            "quality_score": state["quality_score"],
        }
    )
    approved = bool(decision.get("approve", False)) if isinstance(decision, dict) else bool(decision)
    return {
        "approved": approved,
        "log": [f"[human_approval] human decision: {'APPROVED' if approved else 'REJECTED'}"],
    }


def route_after_approval(state: State) -> str:
    return "finalize" if state["approved"] else "cancelled"


def finalize(state: State) -> dict:
    return {
        "final_output": f"[SENT TO CLIENT]\n{state['draft']}",
        "log": ["[finalize] recommendation sent"],
    }


def cancelled(state: State) -> dict:
    return {
        "final_output": "[NOT SENT — human rejected the recommendation]",
        "log": ["[cancelled] human rejected; nothing was sent"],
    }


# ---------------------------------------------------------------------------
# GRAPH ASSEMBLY
# ---------------------------------------------------------------------------
def build_graph(llm, checkpointer=None):
    graph = StateGraph(State)

    graph.add_node("search", make_search_node())
    graph.add_node("draft", make_draft_node(llm))
    graph.add_node("critique", make_critique_node(llm))
    graph.add_node("loop_bookkeeping", increment_retries)
    graph.add_node("human_approval", human_approval)
    graph.add_node("finalize", finalize)
    graph.add_node("cancelled", cancelled)

    graph.add_edge(START, "search")
    graph.add_edge("search", "draft")
    graph.add_edge("draft", "critique")

    graph.add_conditional_edges(
        "critique",
        route_after_critique,
        {"draft": "loop_bookkeeping", "human_approval": "human_approval"},
    )
    graph.add_edge("loop_bookkeeping", "draft")

    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {"finalize": "finalize", "cancelled": "cancelled"},
    )
    graph.add_edge("finalize", END)
    graph.add_edge("cancelled", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


MERMAID_DIAGRAM = """
flowchart TD
    START((START)) --> search
    search --> draft
    draft --> critique
    critique -- score < threshold & retries left --> loop_bookkeeping
    loop_bookkeeping --> draft
    critique -- score OK or retries exhausted --> human_approval
    human_approval -- approved --> finalize
    human_approval -- rejected --> cancelled
    finalize --> END((END))
    cancelled --> END
"""
