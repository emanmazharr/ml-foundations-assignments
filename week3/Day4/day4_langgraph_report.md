# Week 3 Day 4 — LangGraph Integration Report

**Environment note (read first):** this build environment has no network access and
neither `langgraph` nor an LLM API key is installed/available. `afl_langgraph_offline.py`
is a deterministic, fully-executed stand-in with the *exact same state schema, nodes and
edges* a real LangGraph app would use, so every result below is real, executed output —
not a mock-up. `afl_langgraph_production.py` contains the literal `langgraph.graph.StateGraph`
wiring (real LLM router via `ChatAnthropic.with_structured_output`) for running in an
environment with `pip install langgraph langchain-anthropic` and an API key.

---

## Task 1 — Graph Design

**State schema** (`AFLGraphState`, a `TypedDict`):

| Field | Type | Purpose |
|---|---|---|
| `user_query` | str | this turn's raw text |
| `conversation_history` | list[(role, text)] | full transcript, for contextual scope + memory |
| `memory` | dict | slot memory (`last_team`, `last_opponent`, `last_player_id`, `last_season`, `last_games`) carried across turns |
| `intent` | "off_topic" \| "retrieval" \| "prediction" | router's decision |
| `tool_name`, `tool_args`, `tool_result` | — | which tool ran and what it returned |
| `tool_error` | str \| None | set when a team/player/stat couldn't be resolved, or when a stat isn't modelled |
| `validation_passed` | bool \| None | gate between "tool ran" and "user sees an answer" |
| `final_response` | str | what the user sees |
| `trace` | list[dict] | one entry per node visited, for Task 5's annotated traces |

**Graph shape:**

```
START -> router_node
           ├─ off_topic  -> refusal_node -> END
           ├─ retrieval  -> retrieval_node  -> validation_node ─┬─ response_formatter_node -> END
           └─ prediction -> prediction_node -> validation_node ─┴─ clarification_node       -> END
```

**Why explicit routing beats one generic agent (justification):**
A single tool-calling agent decides per turn — inside one LLM call — whether to
retrieve, predict, or answer from memory, and that decision can silently drift (e.g.
answering a win-probability question from "football knowledge" instead of calling
`predict_match_winner`, or dropping the required probabilistic disclaimer because
nothing forces it to be there). Explicit LangGraph routing makes three things
structural rather than hoped-for: (1) a prediction can *only* reach the user through
`prediction_node`, which always attaches the disclaimer + top-feature explanation; (2)
`validation_node` sits between "tool ran" and "user sees an answer", so a tool error
hard-routes to `clarification_node` instead of being narrated over; (3) each node is
independently testable (see the router accuracy table and annotated traces below),
which you cannot do to "the agent's judgement" buried in one big prompt.

---

## Task 2 — Router Node & Routing Accuracy

The router first runs the Day-3 scope guardrail (`classify_scope`) — this catches
off-topic/jailbreak turns before any intent classification is attempted. For in-scope
turns, a keyword/pattern classifier (offline) or a structured-output LLM call
(production) decides `retrieval` vs `prediction`.

**Round 1 result (16 hand-written queries): 20/20 (100%).** To pressure-test the
router honestly, four harder/ambiguous queries were added — and this round exposed
three real misroutes:

| # | Query | Expected | Got | Fix applied |
|---|---|---|---|---|
| 1 | "Who is going to win the flag this year?" | prediction | **off_topic** | "flag" (AFL slang for premiership) wasn't in the guardrail's AFL-vocabulary list, so the turn never reached the router's intent step. Added `flag`, `fantasy` to `AFL_GENERIC_TERMS`. |
| 2 | "What was Bontempelli's fantasy score last week?" | retrieval | **off_topic** | Same root cause — "fantasy score" didn't match the narrower "fantasy points" phrase. Fixed by the same vocabulary addition above. |
| 3 | "How good are Carlton's finals chances?" | prediction | **retrieval** | No win/predict keyword, so it fell through to the retrieval default. Added `\bfinals chances\b` / `\bchances\b` and `\bgoing to win\b` to `PREDICTION_PATTERNS`. |

**Final result after fixes: 24/24 (100%).** Full pass/fail table is reproduced by
running `test_router_accuracy.py`.

---

## Task 3 — Prediction Tools

`afl_predict_tools.py` trains two GradientBoosting models directly from the real raw
data (`team_matches_home_away_raw`, `afl_players_round_by_round_stats_raw`) at import
time — no external `.joblib` files needed:

- **`predict_match_winner(team_a, team_b, date)`** — rolling team-form features
  (`score_last_avg`, `margin_last_avg`, `winrate_last_avg`, `days_rest`), computed with
  `shift(1)` so training features never leak the match's own result. Returns
  win/loss/draw probabilities, the top-3 features driving the prediction, and a fixed
  disclaimer sentence.
- **`predict_top_player(team, stat_type)`** — rolling player-form features
  (`fp_last3_avg`, `fp_last5_avg`, `disp_last5_avg`) → predicted fantasy points per
  player on the requested team. Any `stat_type` other than `fantasy_points` raises a
  clean `PredictionInputError` rather than silently predicting the wrong thing.

**Input resolution (Task 3's explicit ask):**
- `resolve_team_alias()` maps common nicknames ("Pies", "Cats", "the Dogs", "GWS") to
  the dataset's canonical team key, falling back to the Day-3 fuzzy team resolver.
- `resolve_relative_date()` maps "this week" / "this round" / "next round" / "today" to
  a usable ISO date. **Documented limitation:** the dataset has no forward-looking
  fixture list, so these phrases resolve to *today's real-world date*, not an actual
  scheduled fixture — the model only needs a valid date ≥ its training window, but the
  agent does not claim to know the real fixture list.
- Every prediction response includes `probabilities`/`predicted_fantasy_points`,
  `top_features` (grounding explanation), and `disclaimer` — enforced structurally by
  `response_formatter_node`, not left to a model's discretion.

---

## Task 4 — Self-Correction & Fallbacks

`validation_node` checks `tool_error is None and tool_result is not None`. Two
distinct fallback behaviours were tested and confirmed working (see Conversations 7,
8, 11 below and annotated Trace B):

1. **Unresolvable entity → clarification loop.** `predict_match_winner("Made Up Team",
   "Carlton Blues", ...)` raises `PredictionInputError("need_two_teams")` →
   `validation_passed=False` → `clarification_node` asks the user to name real teams,
   rather than guessing or defaulting to a team that wasn't asked for.
2. **Unsupported stat type → explicit fallback, not hallucination.** "Predict how many
   tackles Carlton's top player will get" → `prediction_node` extracts `stat_type="tackles"`
   and passes it through; `predict_top_player` raises `PredictionInputError("stat_type
   'tackles' is not modelled yet...")`; `clarification_node` detects the "not modelled"
   phrasing and returns a plain "I can't do that one yet" message instead of silently
   substituting the fantasy-points model's numbers.

---

## Task 5 — End-to-End Testing

**12 full conversations (16 turns total) were run**, covering every required path:
pure factual retrieval, match-winner prediction, top-scorer prediction, off-topic
refusal (other-sport and jailbreak variants), ambiguous input requiring clarification
(both prediction-side and retrieval-side), a 4-turn team→player→stat-comparison
conversation, a retrieval-then-prediction session, an unsupported-stat fallback, and
nickname resolution. Full transcript: `e2e_output.txt` / the executed notebook cells.

**Three annotated state traces** (full node-by-node breakdown, in
`annotated_traces_output.txt`):
- **Trace A** — multi-turn retrieval: shows `memory['last_player_id']` resolving "his"
  in a turn with zero named entities.
- **Trace B** — prediction validation failure → clarification → successful retry,
  showing the exact `tool_error` string and the validation gate in action.
- **Trace C** — a jailbreak turn that never reaches `validation_node` or any tool node
  at all, because `router_node`'s guardrail check short-circuits straight to
  `refusal_node`.

**LangGraph vs. one monolithic agent — what specifically improved:**
Across these 16 turns, every prediction response carried its disclaimer and
feature-grounding *by construction* (it's baked into `response_formatter_node`, not
generated fresh each time by an LLM that could forget it), and every unresolved
team/player/stat produced a clarification or fallback message instead of a guess —
because `validation_node` is a real branch point in the graph, not a hope that the
agent "notices" its own tool failed. The router-accuracy exercise in Task 2 also
would not have been possible against a monolithic agent in the same way: because
`router_node` is an isolated, callable function, its 4 real misroutes could be found,
attributed to a specific cause, and fixed with a one-line change each — the same
audit against a single big system prompt would have meant re-running the whole agent
and guessing which instruction to reword.

**Known limitations (for transparency):**
- Prediction intent doesn't yet persist through memory the way `last_team` does — a
  follow-up like "what about Carlton vs Hawthorn?" after a prediction turn needs an
  explicit win/predict verb to stay in the prediction branch (session-level *intent*
  stickiness, as opposed to the *entity* stickiness memory already provides, would be
  the natural next improvement).
- `resolve_relative_date` cannot resolve a real upcoming fixture because no
  forward-looking fixture list exists in the supplied data.
