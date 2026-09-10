# Week 2 · Day 4 — CrewAI: Multi-Agent Collaboration, Roles & Task Delegation

A 3-agent crew (Data Analyst → Market Strategist → Report Writer) that
reviews a laptop product catalog, generates customer-segment insights,
and writes a stakeholder-ready summary for a back-to-school marketing
campaign — built both as `Process.sequential` and `Process.hierarchical`.

## Files

| File | Purpose |
|---|---|
| `crewai_agent.ipynb` | Main deliverable — all 5 tasks. **Must be run** in Colab to produce real output before submission. |
| `crew_tools.py` | `ListProductsTool` (Analyst only) and `PriceValueCalculatorTool` (Strategist only). |
| `crew_agents.py` | The 3 agents + the hierarchical crew's manager persona, with role/goal/backstory and role-appropriate tool assignment. |
| `crew_tasks.py` | Task definitions for both the sequential (agent-assigned) and hierarchical (unassigned, manager-delegated) crews, including the documented format-mismatch fix. |
| `crew_build.py` | `build_sequential_crew()` and `build_hierarchical_crew()`. |
| `products.json` | Reused laptop catalog from earlier days. |
| `requirements.txt` | `pip install -r requirements.txt` |
| `WRITEUP_day4.md` | 1-page write-up: design rationale, sequential-vs-hierarchical table, token usage, success criteria, and the "was multi-agent worth it" analysis. |

## How to run (Google Colab)

1. Upload `crewai_agent.ipynb` plus all `.py` files and `products.json`
   into the same Colab session.
2. Use the same free Gemini API key from earlier days.
3. **Runtime → Run all.**
4. Download the executed notebook and submit alongside the other files.

## Why `gemini-3.5-flash-lite` (not `gemini-3.6-flash`)

CrewAI's multi-agent, multi-task runs make far more LLM calls per run
than a single agent — a 3-agent sequential crew makes at least 3 calls,
and the hierarchical version makes more still for the manager's own
delegation/review reasoning. The "flash-lite" tier's free-tier quota is
substantially higher than the full `gemini-3.6-flash` model's, which is
necessary headroom to run both crew versions today without hitting a
quota wall. (Current free-tier limits change over time — check
https://ai.google.dev/gemini-api/docs/rate-limits before relying on a
specific number.)

> **Note:** this project originally used `gemini-2.0-flash-lite`, which
> Google retired on 2026-07-21 in favor of the 3.5 line. If you see a
> `404 ... model ... no longer available` error, it means the code is
> still pointing at a retired model name — update `get_llm()`'s default
> in `crew_agents.py` to whatever model ID Google's docs currently list.
