"""
afl_langgraph_production.py
------------------------------
The REAL LangGraph wiring for this app: same nodes, same state schema,
same edges as afl_langgraph_offline.py, expressed with the actual
`langgraph` library. This is what to run outside this sandbox.

    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...      # or GOOGLE_API_KEY for Gemini
    python afl_langgraph_production.py

The router node here uses a small structured-output LLM call (Task 2's
"can be a small LLM call with structured output" option) instead of the
offline regex router, since a real LLM is available in this environment.
Everything downstream (tool nodes, validation, clarification, response
formatting) is IDENTICAL logic to afl_langgraph_offline.py - only the
router's intent-classification step differs.
"""

from typing import TypedDict, List, Tuple, Optional, Any, Dict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from afl_guardrails import classify_scope, refusal_response, SYSTEM_PROMPT
import afl_data_tools as dt
import afl_predict_tools as pred

# reuse the exact same node bodies as the offline version for
# retrieval/prediction/validation/clarification/response-formatting -
# only the router node is swapped for a real LLM call below.
from afl_langgraph_offline import (
    AFLGraphState, retrieval_node, prediction_node, validation_node,
    clarification_node, response_formatter_node, refusal_node,
)


class RouterDecision(BaseModel):
    intent: Literal["retrieval", "prediction", "off_topic"] = Field(
        description="retrieval = factual/stat lookup; prediction = asks who "
                    "will win or who will top-score; off_topic = not about AFL"
    )


_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
_router_llm = _llm.with_structured_output(RouterDecision)


def llm_router_node(state: AFLGraphState) -> AFLGraphState:
    text = state["user_query"]
    context = [x for role, x in state["conversation_history"] if role == "user"][-4:]
    scope = classify_scope(text, context)   # keep the deterministic guardrail as a fast first pass
    state.setdefault("trace", [])
    step = {"node": "llm_router_node", "input": text, "scope_classification": scope}

    if not scope["in_scope"]:
        state["intent"] = "off_topic"
        step["decision"] = "off_topic"
        state["trace"].append(step)
        return state

    decision = _router_llm.invoke([
        SystemMessage(content="Classify this AFL chat message's intent."),
        HumanMessage(content=text),
    ])
    state["intent"] = decision.intent
    step["decision"] = decision.intent
    state["trace"].append(step)
    return state


def route_after_router(state: AFLGraphState) -> str:
    return {"off_topic": "refusal", "prediction": "prediction", "retrieval": "retrieval"}[state["intent"]]


def route_after_validation(state: AFLGraphState) -> str:
    return "respond" if state["validation_passed"] else "clarify"


def build_graph():
    graph = StateGraph(AFLGraphState)
    graph.add_node("router", llm_router_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("prediction", prediction_node)
    graph.add_node("refusal", refusal_node)
    graph.add_node("validation", validation_node)
    graph.add_node("clarify", clarification_node)
    graph.add_node("respond", response_formatter_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_after_router,
                                 {"off_topic": "refusal", "prediction": "prediction", "retrieval": "retrieval"})
    graph.add_edge("retrieval", "validation")
    graph.add_edge("prediction", "validation")
    graph.add_conditional_edges("validation", route_after_validation,
                                 {"respond": "respond", "clarify": "clarify"})
    graph.add_edge("refusal", END)
    graph.add_edge("respond", END)
    graph.add_edge("clarify", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    memory: Dict[str, Any] = {}
    history: List[Tuple[str, str]] = []
    for q in [
        "How is Carlton Blues going lately?",
        "Who will win if Carlton plays Collingwood this week?",
        "What's the weather today?",
    ]:
        state = app.invoke({"user_query": q, "conversation_history": history, "memory": memory, "trace": []})
        print("USER:", q)
        print("AGENT:", state["final_response"])
        history.append(("user", q))
        history.append(("assistant", state["final_response"]))
