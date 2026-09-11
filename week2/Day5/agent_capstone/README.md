# ClientOnboard AI — Week 2 Day 5 Capstone

A production-style client-onboarding agent for a freelancing/web-development business.

## What it does
1. Validates a new client request.
2. Looks up service/package information from a local JSON data source.
3. Classifies the request and recommends a suitable service.
4. Produces a structured onboarding summary.
5. Stops at a human approval checkpoint before a consequential action (sending/confirming a proposal).
6. Exposes the agent through FastAPI.
7. Logs tool calls, latency, errors and run metadata.

## Architecture
Input -> Validate -> Service Lookup -> Analyze -> Draft Proposal -> Human Approval -> Final Output

## Run
```bash
pip install -r requirements.txt
uvicorn api:app --reload
```

Open:
- http://127.0.0.1:8000/docs

API:
POST `/onboard`

Example body:
```json
{
  "client_name": "Ali",
  "email": "ali@example.com",
  "project_description": "I need a responsive bakery website with products and an online order form.",
  "budget": 1200,
  "deadline_days": 20
}
```

To approve a proposal:
POST `/approve/{run_id}` with:
```json
{"approved": true}
```

## External data source
`data/services.json` is the external/local business knowledge source used by the service lookup tool.

## Failure handling
- Bad/missing input -> HTTP 422 validation error.
- Tool/data failure -> graceful fallback message and logged error.
- Model refusal is represented by a safety guard that refuses requests for harmful/illegal activity.

## Evaluation
See `evaluation.csv` and the executive report PDF.
