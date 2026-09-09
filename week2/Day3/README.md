# Week 2 · Day 3 — LangGraph: Stateful, Multi-Step & Cyclical Agent Workflows

A self-correcting product-recommendation research assistant built as a
LangGraph `StateGraph`: search → draft → critique → (loop back to draft,
or forward to a human-approval interrupt) → finalize/cancel.

## Files

| File | Purpose |
|---|---|
| `langgraph_agent.ipynb` | Main deliverable — all 5 tasks. **Must be run** in Colab (see below) to produce real output before submission. |
| `lg_state.py` | The shared `State` TypedDict + `CritiqueResult` structured-output schema (Task 1). |
| `lg_tools.py` | Plain product-catalog lookup used by the `search` node. |
| `lg_graph.py` | The full graph: nodes, the conditional self-correction loop, the `human_approval` interrupt, and `build_graph()` (Tasks 2–5). |
| `products.json` | Local JSON "database" of laptop prices (reused concept from Day 2). |
| `requirements.txt` | `pip install -r requirements.txt` |
| `WRITEUP_day3.md` | 1-page write-up: graph diagram, conditional-loop rationale, human-in-the-loop discussion, AgentExecutor vs. LangGraph comparison. |

## Graph diagram

See the ASCII diagram in Task 1 of the notebook, or the Mermaid diagram in
`lg_graph.py`'s `MERMAID_DIAGRAM` string / `WRITEUP_day3.md`. To render
LangGraph's own auto-generated diagram inside the notebook (optional,
needs internet inside Colab):
```python
from IPython.display import Image
Image(graph.get_graph().draw_mermaid_png())
```

## How to run (Google Colab — same flow as Day 2)

1. Go to https://colab.research.google.com, upload `langgraph_agent.ipynb`.
2. Upload `lg_state.py`, `lg_tools.py`, `lg_graph.py`, and `products.json`
   into the same Colab session folder.
3. Use the same free Gemini API key from Day 2 (or get one at
   https://aistudio.google.com/app/apikey — no billing required).
4. **Runtime → Run all.** The API key prompt uses `getpass`, so it is
   never saved into the notebook file itself.
5. Once finished, **File → Download → .ipynb** to get the version with
   real executed output, and submit that alongside the other files.

## Why `gemini-3.6-flash`

Google has retired `gemini-2.5-flash` for new users and now points
everyone to `gemini-3.6-flash` (as of its own API error message). Unlike
Day 2's `AgentExecutor` + `RunnableWithMessageHistory` combination — which
broke on Gemini 3.6's "thinking"-enabled structured content blocks — this
Day 3 graph never round-trips raw model messages through LangChain's
message-history serialization, so that specific failure doesn't apply
here. As a defensive precaution anyway, `lg_graph.py`'s `_extract_text()`
helper safely pulls plain text out of a response whether `.content` comes
back as a string or as a list of content blocks, so `draft`/`critique`
text stays clean either way.
