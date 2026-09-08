# Write-up — Week 2 / Day 2: LangChain vs. Raw-Python Agent

## 1. Concept mapping

| Day 1 (raw Anthropic API) | LangChain equivalent |
|---|---|
| `anthropic.Anthropic()` client | `llm` — a `BaseChatModel` (here `ChatGoogleGenerativeAI`) |
| Hand-written JSON tool schemas + a dispatch dict | `@tool`-decorated functions — docstring + type hints generate the schema |
| `run_agent()` while-loop | `AgentExecutor` |
| `messages` list, appended by hand every turn | `RunnableWithMessageHistory` (or classic `ConversationBufferMemory`) |
| `working_memory` scratchpad + manual `print()` logging | `AgentExecutor(verbose=True)` trace / callbacks |
| Manually built `tool_result` dict | Built internally by `AgentExecutor` |

## 2. LCEL and the pipe operator

`prompt | llm | StrOutputParser()` chains three objects that all implement
the same `Runnable` interface (`.invoke()`, `.stream()`, `.batch()`).
Python's `|` is overloaded to compose them into one `RunnableSequence`, so
`chain.invoke(x)` is exactly equivalent to
`StrOutputParser().invoke(llm.invoke(prompt.invoke(x)))`. It is function
composition with a readable syntax, not a new execution model — everything
we built by hand in `run_agent()`'s single-call path on Day 1 collapses
into this one line.

## 3. Annotated reasoning trace (Task 3)

Running `agent_executor.invoke(...)` with `verbose=True` on the task
*"Look up the price of the UltraBook Pro and the BudgetBook Lite, then
tell me which is cheaper"* produces a trace with the same three phases as
Day 1's hand-logged loop:

- **REASON** — `AgentExecutor` enters its chain and prints the model's
  chosen action, e.g. `Invoking: get_product_price` with
  `{"product_name": "UltraBook Pro"}`. Directly equivalent to Day 1's
  `[model reasoning/text]` / tool-choice log line.
- **ACT** — the real `get_product_price` Python function executes with
  those arguments. Equivalent to Day 1's `[ACT] calling tool '...'` line.
- **OBSERVE** — the returned JSON string (price + specs) is printed and
  automatically appended to the model's context before the next
  reasoning step. Equivalent to Day 1's `[OBSERVE] result: ...` line.

This repeats for the second product before the model emits a
`Final Answer:` — the same two-sequential-tool-call pattern as Day 1's
Lahore/Karachi weather comparison.

**Similar to Day 1:** the control flow is identical — reason, act,
observe, repeat, until the model stops requesting tools.

**Hidden now:** the exact `tool_result` block shape, how the model's raw
tool-call output is parsed, and how prior actions get re-serialized into
the next prompt (the "agent scratchpad") are handled entirely inside
`AgentExecutor` / `create_tool_calling_agent`. On Day 1 every one of those
fields was visible and editable in plain Python; here they still exist,
but only surface through `verbose=True` output or custom callbacks.

## 4. Memory (Task 4)

`RunnableWithMessageHistory`, keyed by `session_id`, lets a 3-turn
conversation work correctly: *"Find the price of the UltraBook Pro"* →
*"Now compare it to the BudgetBook Lite"* → *"Which one should I recommend
to a budget-conscious client?"*. Turn 2 only resolves "it" correctly
because the model can see Turn 1's tool result in `chat_history`; Turn 3
depends on both prior turns. This is the same growing-transcript idea as
Day 1's `messages` list, except LangChain manages appending to it
automatically instead of requiring a manual `messages.append(...)` after
every turn.

## 5. Structured output & error handling (Task 5)

`llm.with_structured_output(Recommendation)` forces a final answer into a
Pydantic schema (`recommended_product`, `price_usd`,
`price_difference_usd`, `reasoning`); LangChain generates the schema
instructions for the model and validates/parses the response back into a
real Python object, raising a validation error if the shape doesn't match.

For error handling, `get_product_price` raises a plain `KeyError` with a
descriptive message for an unknown product — deliberately kept consistent
with Day 1's "never fail silently" rule. `AgentExecutor` catches that
exception internally, turns it into an error string, and feeds it back to
the model as the tool's observation instead of crashing the chain; the
model then explains the lookup failed rather than inventing a price. The
only extra configuration needed beyond the tool's own exception was
`handle_parsing_errors=True` on `AgentExecutor`, which additionally covers
the case where the model's own tool-call formatting is malformed rather
than the tool itself failing.

## 6. What LangChain made easier vs. what got "leaky"

LangChain removed nearly all of Day 1's manual plumbing: tool schemas
became docstrings, the reason/act/observe loop became `AgentExecutor`,
and multi-turn memory became a two-line wrapper instead of a hand-managed
list. The trade-off shows up exactly where the mapping table above ends:
the precise mechanics of how a tool exception gets reformatted into model
input, how the "agent scratchpad" is reconstructed each turn, and how
`create_tool_calling_agent` decides which of a model's outputs count as a
tool call are no longer things we wrote — they're implemented inside
LangChain, correct by default, but a black box unless you go read the
library's source when something behaves unexpectedly. Having built the
loop by hand on Day 1 makes that black box much less mysterious.
