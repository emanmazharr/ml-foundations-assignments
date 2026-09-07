# Week 2 · Day 1 — Agent Foundations: Reasoning Loops, Tool Calling & Raw Python Agents

A minimal, from-scratch ReAct agent built directly on the Anthropic Messages
API — no LangChain, no LangGraph. Built as part of an internship agentic-AI
onboarding curriculum.

## 📂 What's in this repo

| File | Purpose |
|---|---|
| **`agent_foundations.ipynb`** | 📓 Main deliverable — the full, **already-executed** notebook covering all 5 tasks, with real printed output for every cell. Open this first. |
| **`WRITEUP.md`** | 1-page write-up: the ReAct loop, tool schemas used, and failure modes observed + mitigations. |
| `agent.py` | Core module: tool JSON schemas, tool implementations, the `run_agent()` ReAct loop, and a `MockAnthropicClient` that mimics the real Anthropic SDK interface so the loop is testable without an API key. |
| `demos.py` | Scripted scenarios: the multi-tool-call happy path (Task 3) and five deliberate failure modes (Task 5). |
| `live_demo.py` | Runs the identical `run_agent()` loop against the **real** Claude model — requires `ANTHROPIC_API_KEY`. |
| `build_notebook.py` | Script that programmatically generates `agent_foundations.ipynb` (kept for reproducibility/transparency). |

## ✅ Task checklist

- [x] **Task 1** — Agent vs. chatbot vs. workflow, what makes something agentic, ReAct diagram + pseudocode, when an agent is overkill → `agent_foundations.ipynb`, Task 1 section.
- [x] **Task 2** — 3 tools defined with full JSON schemas (`calculator`, `get_weather`, `read_text_file`), explanation of why descriptions matter, one manual tool-use request/response cycle → `agent.py` (`TOOLS`), notebook Task 2 section.
- [x] **Task 3** — Full `while`-loop agent (reason → check tool_use → execute → append tool_result → repeat), `max_iterations` safeguard, tested on a 2+ tool-call task (compare weather in two cities) → `agent.py` (`run_agent`), notebook Task 3 section.
- [x] **Task 4** — Conversation memory vs. working memory explained and implemented separately; live step-by-step logging of every reasoning step, tool call, and observation → notebook Task 4 section.
- [x] **Task 5** — Agent deliberately broken 5 ways (infinite loop, hallucinated tool, wrong arguments, tool error, ambiguous request), each documented with a mitigation; closing paragraph on why frameworks like LangChain/LangGraph/CrewAI exist → notebook Task 5 section + `WRITEUP.md`.

## ▶️ How to run it yourself

The notebook runs **out of the box, no API key required** — it uses a
scripted `MockAnthropicClient` that mimics the real Anthropic SDK response
shape, so the exact same `run_agent()` loop, logging, memory handling, and
error handling can be verified deterministically and reproducibly (e.g. in
CI or by a grader with no secrets configured).

```bash
pip install anthropic nbformat nbclient ipykernel
python3 build_notebook.py    # (re)generates agent_foundations.ipynb
jupyter nbconvert --to notebook --execute --inplace agent_foundations.ipynb
# or just open agent_foundations.ipynb directly — outputs are already saved
```

To see the **real Claude model** make these decisions instead of the mock
script:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python3 live_demo.py "Look up the weather in Lahore and Karachi and tell me which is warmer."
```

## 🧠 Design note

`MockAnthropicClient` implements the identical `.messages.create(...)`
interface as `anthropic.Anthropic().messages`, returning objects with the
same `.content` / `.stop_reason` shape. This means `run_agent()` is
**completely unaware** of whether it's talking to the real model or a
scripted stand-in — the same production loop code is what gets tested in
both mock and live mode, rather than maintaining two different code paths.
