# Production Monitoring Checklist

Track:
- Error rate: validation errors, tool failures, server errors.
- Latency: p50, p95 and p99 response time.
- Cost: model tokens and estimated cost per run.
- Tool reliability: lookup failures/timeouts.
- Output quality: recommendation accuracy and human approval rate.
- Safety: refusal rate and blocked harmful requests.
- Drift: changes in request types, budgets and successful outcomes.

Suggested alerts:
- Error rate > 5% over 15 minutes.
- p95 latency > 3 seconds for 15 minutes.
- Tool failure rate > 2%.
- Estimated cost/run increases > 25% week-over-week.
- Human rejection rate > 30% over a rolling 100 runs.
- Safety incidents: immediate alert for any confirmed harmful output.

Cadence:
- Daily automated dashboards.
- Weekly quality review.
- Monthly evaluation against a fixed regression set.
- Re-run the full 8+ case evaluation whenever prompts, tools, model, or business rules change.
