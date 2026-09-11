# Architecture

```text
Client
  |
  v
FastAPI /onboard
  |
  v
Input Validation
  |
  v
Safety Guard
  |
  v
Service Lookup Tool ----> data/services.json
  |
  v
Analysis / Recommendation Node
  |
  v
Proposal Draft Node
  |
  v
HUMAN APPROVAL CHECKPOINT
  |                    |
 approve              reject
  |                    |
  v                    v
Final proposal      Rejected
  |
  v
Structured JSON + Logs
```

### Framework choice
This version is implemented as a **raw deterministic pipeline** (plain Python functions called in sequence), not LangGraph or CrewAI. That was a deliberate v1 choice, not an oversight: the flow is a single linear path with one branch point (safety refusal) and one pause point (human approval) — no cyclical reasoning, no multiple autonomous roles negotiating a shared goal, and no LLM call to orchestrate. A raw pipeline is the cheapest, fastest, and most testable way to satisfy that shape; every step is a pure function with an exact expected output, which is what makes the evaluation harness fully reproducible (see `evaluation_runner.py`).

LangGraph becomes the right choice once this system needs things a raw pipeline can't express cleanly: multiple branches based on LLM judgment rather than fixed rules, retries/self-correction loops, or persisting/resuming a paused run across process restarts (today, `APPROVALS` is in-memory and lost on restart). CrewAI was not selected because the problem is one job end-to-end, not several specialist roles collaborating — the analyzer, drafter, and safety check don't need separate agent identities. FastAPI provides a clean production-facing API boundary in either case.
