# Monitoring & Maintenance Checklist

| Signal | Target / alert | Cadence | Action |
|---|---|---|---|
| Response latency p95 | Alert > 3 s; critical > 8 s | Daily | inspect tool/model latency; reduce payload or timeout |
| Tool error rate | Alert > 5% weekly | Daily/weekly | inspect failing tool/entity resolution and dataset freshness |
| Off-topic leak rate | Alert > 1% of guardrail suite | Weekly | add adversarial examples and tighten scope rules |
| Prediction accuracy drift | Alert when 4-week accuracy falls >5 pp vs baseline | After each round | retrain/evaluate on time-based holdout |
| Prediction calibration | Monitor Brier/log loss | After each round | recalibrate or retrain if degraded |
| Gemini quota/errors | Alert on repeated 429/5xx | Daily | backoff, rate-limit, review model quota |
| Grounding failures | Any stat not traceable to a tool result | Every eval run | block response and fix formatter/tool routing |

## Weekly refresh loop
1. Ingest newly completed match results into the team-match and player round-by-round tables.
2. Validate schema, duplicates, missing values, team/player IDs, and match dates.
3. Recompute rolling-form features using only information available before each prediction (shifted features).
4. Run the 30+ evaluation suite and compare with the prior report.
5. Retrain the match-winner and player models after each completed round **when enough new labeled matches exist**; otherwise refresh features only.
6. Keep the previous model as rollback; record model version, training cutoff date, metrics, and dataset version.
7. Promote the new model only if time-based holdout metrics and sanity checks do not regress.
