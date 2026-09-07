# Write-up — Week 2 / Day 1: Agent Foundations

## 1. The ReAct loop, in practice

The agent implemented in `agent.py` (`run_agent()`) follows the classic
**Reason → Act → Observe → repeat** pattern directly against the Anthropic
Messages API:

1. **Reason** — the full conversation so far is sent to the model along
   with the list of available tools. The model either returns plain text
   (it's done) or one or more `tool_use` blocks (it wants to act).
2. **Act** — for each `tool_use` block, the corresponding Python function is
   executed with the model-supplied arguments.
3. **Observe** — the result (or error) of that execution is packaged as a
   `tool_result` block and appended to the conversation as the next "user"
   turn, exactly as the Anthropic API expects tool results to be returned.
4. **Repeat** — the loop goes back to step 1 with the updated conversation,
   until either the model stops requesting tools, or a hard
   `max_iterations` safeguard is hit (set to 6–8 in the demos), at which
   point the loop stops itself and reports `stopped_reason:
   "max_iterations_reached"` rather than hanging indefinitely.

This loop is genuinely agentic because *the model decides*, turn by turn,
whether it needs another tool call or has enough information to answer —
the developer never hardcodes "call weather, then call weather again, then
answer." On the multi-city weather comparison task, the model correctly
chained two separate tool calls before producing a final answer, which a
single non-agentic prompt could not have done (it would have had to guess
both temperatures).

Two kinds of memory were kept deliberately separate: **conversation
memory** (`messages`, the verbatim transcript sent back to the model every
turn) versus **working memory** (`working_memory`, a scratchpad the agent
code — not the model — maintains, listing every tool call and observation
for logging/debugging purposes). Every reasoning snippet, tool call, and
observation is printed live during the run, which is the debugging habit
this exercise was meant to instill before frameworks abstract it away.

## 2. Tool schemas used

Three tools were registered, each with a `name`, a `description`, and a
JSON-Schema `input_schema`:

- **`calculator`** — `{expression: string}` → evaluates basic arithmetic.
- **`get_weather`** — `{city: string}` → looks up temperature/condition
  from a small fixed table (stub for a real weather API).
- **`read_text_file`** — `{filename: string}` → reads a local `.txt` file.

Descriptions were written to state not just *what* the tool does but *when*
to use it, the expected input format, and what happens on failure (e.g.
"returns an error if the city isn't found — don't guess a number"). This
matters because the description and schema are the model's *only* source of
truth about a tool; vague descriptions are the single biggest cause of
wrong-tool selection and malformed arguments in practice.

## 3. Failure modes observed

Five failure modes were deliberately triggered (via scripted mock model
responses, so each is reproducible on demand) and one mitigation applied
to each:

| Failure mode | Mitigation |
|---|---|
| Infinite / repetitive tool-calling loop | Hard `max_iterations` cap that stops the loop and returns a "best-effort" result instead of hanging. |
| Hallucinated tool call (model invents a tool that was never registered) | `execute_tool()` checks the name against a registry and returns a structured `is_error=True` result instead of crashing. |
| Wrong / malformed tool arguments | Input validation inside the tool itself, surfaced back to the model as an error `tool_result` so it can retry with corrected arguments (it did, successfully). |
| Tool executes but returns a domain-level error (e.g. unknown city) | Tool raises a descriptive exception; a generic `try/except` around every tool call reports it as `is_error=True` rather than letting the process crash. |
| Silently swallowed errors | Never return an empty string on failure — always return a descriptive message and set `is_error=True`, so the model treats it as a real (negative) observation rather than assuming success. |
| Ambiguous user request (missing required info, e.g. no city given) | Clear tool `input_schema`/`required` fields plus a system prompt instructing the model to ask a clarifying question rather than guessing a default. |

## 4. Why frameworks exist, having just built this by hand

The ~150-line loop above already had to handle message formatting, tool
result schemas, error propagation, and a loop guard correctly. A real
production agent additionally wants: context-window trimming/summarization
of long conversation memory, retry/backoff on transient API errors,
streaming responses, parallel tool execution, multi-agent handoffs, and
built-in tracing/observability — plus a shared vocabulary so different
engineers don't each reinvent this loop slightly differently. Frameworks
like LangChain, LangGraph, and CrewAI exist to package those recurring,
easy-to-get-subtly-wrong pieces into reusable primitives. The tradeoff is
that they also hide the loop itself behind an abstraction — which is
exactly why building it by hand first is valuable: when a framework's
executor or graph node misbehaves tomorrow, the underlying mechanics (what
a `tool_use` block looks like, how a `tool_result` gets threaded back in,
where a loop guard needs to live) will already be familiar rather than a
black box.
