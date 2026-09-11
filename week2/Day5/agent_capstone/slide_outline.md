# 5–7 Minute Stakeholder Presentation

## Slide 1 — Problem & Goal (45 sec)
- Freelance teams receive unstructured client requests.
- Manual triage takes time and can cause inconsistent responses.
- Goal: automate first-level onboarding while keeping consequential decisions human-approved.

## Slide 2 — Solution (45 sec)
- ClientOnboard AI validates, categorizes, recommends a service and drafts a proposal.
- Output is structured JSON for easy integration with CRM/frontend systems.

## Slide 3 — Architecture (60 sec)
- FastAPI → Validation → Safety → Service Lookup → Analysis → Proposal → Human Approval.
- Local JSON service catalog is the external data source.
- Logs capture run ID, tool calls, latency and errors.

## Slide 4 — Why This Architecture? (45 sec)
- Deliberate v1 choice: a raw deterministic pipeline, not LangGraph or CrewAI.
- The flow is one linear path with one branch and one pause point — no multi-role
  collaboration, no cyclical reasoning, so a lighter framework earns its keep.
- LangGraph becomes the right call once we add self-correction loops or
  persist paused runs across restarts (see Next Steps).

## Slide 5 — Evaluation (75 sec)
- 8 test cases, 2 of them edge/adversarial, scored programmatically from real agent output.
- Criteria: success, accuracy, tone, safety, latency, cost — 100% task success on this suite.
- Manual review of the drafts surfaced a pattern the pass/fail metrics don't catch:
  scope ambiguity on short requests (see next slide's fix).
- Invalid input was rejected; harmful request was refused.

## Slide 6 — Human-in-the-Loop & Safety (45 sec)
- Agent stops before consequential proposal approval.
- Human decides approve/reject.
- Safety guard blocks credential theft and other harmful requests.

## Slide 7 — Limitations & Next Steps (60 sec)
- Current version uses local JSON and in-memory approval.
- Next: database, authentication, persistent audit logs, live CRM integration, stronger evaluation, retries and production LLM monitoring.
