# Write-up — Week 2 / Day 4: CrewAI

## 1. Task decomposition & role design

The business task — review a laptop catalog, generate customer-segment
insights, write a stakeholder-ready summary — was split into 3
non-overlapping roles:

| Agent | Role | Goal | Backstory |
|---|---|---|---|
| **Data Analyst** | Fact retrieval only | Extract precise facts from the catalog; no opinions, no rounding | A meticulous analyst who's seen bad decisions come from "remembered" prices instead of checked ones |
| **Market Strategist** | Interpretation only | Turn facts into 2-3 customer-segment recommendations, each backed by a computed value metric | Always grounds a recommendation in a specific number, not intuition |
| **Report Writer** | Synthesis only | Produce a polished, under-200-word executive summary | Reorganizes and polishes, never invents new facts |

Each boundary is enforced twice over: in the goal/backstory text, and in
tool access — only the Analyst can touch raw data, only the Strategist
can compute a value metric, and the Writer has no tools at all.

**Why specialization might help, and where it doesn't:** each role
guards a different failure mode — the Analyst must never hallucinate a
number, the Strategist must never skip a segment, the Writer must never
bury the recommendation in jargon. A single generalist juggling all three
concerns in one prompt is more likely to blend "just report facts" with
"now be persuasive" and quietly round a number while trying to sound
compelling. This is **not** worth the added latency/cost if the task is
small and well-bounded, which this specific instance largely is — see
section 6.

## 2. Tools, assigned role-appropriately

- **Data Analyst** → `list_products` only (sole source of ground-truth data).
- **Market Strategist** → `price_value_calculator` only (computes a real
  price-per-GB-RAM number instead of guessing a ratio).
- **Report Writer** → no tools at all — a deliberate choice, since giving
  it tool access would let it re-fetch or re-compute something
  inconsistently with what was already reviewed upstream.

## 3. Tasks, process, and the format-mismatch fix

Three `Task` objects wired with `context=[...]` dependencies so each
later task automatically receives earlier tasks' outputs.

**The fix, documented before it caused a downstream failure at runtime:**
an early draft of the Analyst's `expected_output` just said "list the
products and their prices" — loose enough that the model could paraphrase
a number ("around \$1,500" instead of `1499`), which would break the
Strategist's `price_value_calculator` call, since dividing a paraphrased
string isn't a valid arithmetic expression. The shipped `expected_output`
explicitly forbids rounding/paraphrasing and requires the exact
`price_usd` as a plain integer — a concrete case of a downstream tool's
literal parsing needs dictating an upstream task's `expected_output`
wording, not just its `description`.

## 3b. Environment note: `kickoff()` → `kickoff_async()` in Colab

Running `crew.kickoff()` directly inside a Colab cell raised:

```
RuntimeError: Agent execution was invoked synchronously from within a running
event loop. Use `agent.kickoff_async()` / `crew.kickoff_async()` ...
```

Colab/Jupyter cells already run inside the kernel's own active event loop, and
this CrewAI version's synchronous `kickoff()` path refuses to start a second,
nested loop on top of it — a plain `.py` script has no such pre-existing loop,
so the bug only shows up in a notebook. The fix was to call
`await crew.kickoff_async()` instead (notebooks support top-level `await`),
since `kickoff_async()` runs the crew in a background thread instead of trying
to reuse the kernel's loop. `.raw` and `.token_usage` on the returned
`CrewOutput` work identically either way — this was purely an execution-path
fix, not a change to any prompt or task logic.

## 4. Sequential vs. hierarchical

| | Sequential | Hierarchical |
|---|---|---|
| **Pros** | Predictable, fixed order; cheapest (no manager overhead); easiest to debug | Manager can re-route or reject sub-agent output before it propagates; better for open-ended tasks where the right agent/order isn't fixed in advance |
| **Cons** | No review step — a bad Task 1 output silently flows through | Extra LLM calls for the manager's own delegation + review reasoning → higher token usage/latency for the same work |
| **When to use** | The task structure and agent-to-task mapping is already known and fixed (like this one) | The task is more open-ended, sub-agent quality is variable, or an automated review/rejection gate is specifically wanted |

## 5. Cost, evaluation, and comparison to Day 3's LangGraph agent

Token usage for both runs is captured live via `result.token_usage`
(`prompt_tokens`, `completion_tokens`, `total_tokens`,
`successful_requests`) in the notebook. Real captured numbers from this
run:

| Run | Prompt tokens | Completion tokens | Total tokens | Successful requests |
|---|---|---|---|---|
| Sequential crew | 9,306 | 1,728 | 11,034 | 18 |
| Hierarchical crew | 61,448 | 13,912 | 75,360 | 76 |

The hierarchical run's manager overhead (delegation + review reasoning
per task) cost roughly **6.8x** the total tokens and **4.2x** the number
of requests of the sequential run, for the same underlying work — a
concrete, not just directional, confirmation of the "extra LLM calls"
tradeoff in the pros/cons table above.

Day 3's single-agent LangGraph solution solved a narrower, related task
(compare 2 named laptops for one client) in 2 LLM calls per pass with no
manager overhead — its real captured output: *"For a budget-conscious
client, the BudgetBook Lite is the better choice. At \$549, it is
significantly less expensive than the UltraBook Pro..."* That's not a
direct apples-to-apples comparison (narrower task vs. today's full-catalog
segmentation), but it's illustrative of the general pattern: a
single-purpose agent needs far fewer calls to fully solve a narrow task
than a multi-agent crew needs to solve a broader one.

**3 success criteria**, manually scored 1 (fails) / 2 (partial) / 3 (fully
meets) after reading each run's real captured output:

| Criterion | What it checks |
|---|---|
| Factual grounding | Every price/spec cited matches `products.json` exactly |
| Completeness | All 4 products considered, all segments covered, one hero product named |
| Tone | Reads as a polished stakeholder summary, not an internal data dump |

| Run | Factual grounding | Completeness | Tone |
|---|---|---|---|
| Sequential crew | 3 — Budgetbook Lite ($549/8GB) and Ultrabook Pro ($1499/16GB) prices/specs and the $68.63 and $93.69 price-per-GB-RAM figures all match `products.json` exactly | 2 — 3 segments covered and one hero product named, but Workstation Max never appears in the final output, so not all 4 catalog products were considered | 3 — reads as a clean, polished stakeholder summary ending in one clearly labeled hero pick |
| Hierarchical crew | 3 — Budgetbook Lite, Aircase 13, and Workstation Max prices/specs and their $68.63 / $124.88 / $87.47 price-per-GB-RAM figures all match `products.json` exactly | 2 — 3 segments covered and one hero product named, but Ultrabook Pro never appears in the final output, so (like the sequential run) not all 4 products were considered | 3 — reads as a clean executive summary, comparable in polish to the sequential run |
| Day 3 LangGraph (single-agent) | 3 | 2 — solved its own narrower task fully, not designed for full-catalog segmentation | 3 |

**Observation:** both crew runs independently dropped one product from the final narrative (Sequential omitted Workstation Max; Hierarchical omitted Ultrabook Pro) even though the Analyst's raw catalog facts always contained all 4 — worth noting since neither process, on its own, guarantees every product survives into the final summary. The two runs also picked **different hero products** (Ultrabook Pro vs. Budgetbook Lite) from the same catalog and the same underlying instructions, which is itself a useful reliability data point: the manager-mediated hierarchical run isn't necessarily more consistent, just differently shaped.

## 6. Was the multi-agent crew worth it here?

For this specific task, splitting fact-retrieval, strategy, and writing
into three agents is defensible but not obviously necessary: the task is
bounded and well-understood enough that a single, carefully prompted
agent — given both the catalog tool and the calculator, with a clear
"facts first, then reasoning, then a polished summary" system prompt —
could very plausibly hit all 3 success criteria in one or two LLM calls,
at a fraction of the sequential crew's token cost and without the
hierarchical version's manager overhead at all. The multi-agent structure
earns its cost more clearly as the task grows — more data sources, more
customer segments, or a genuine need for an automated review gate before
a stakeholder sees the output — none of which this particular instance
of the task actually required.
