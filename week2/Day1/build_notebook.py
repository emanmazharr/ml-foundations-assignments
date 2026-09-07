import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# =============================================================================
md(r"""
# Week 2 · Day 1 — Agent Foundations
### Reasoning Loops, Tool Calling & Raw Python Agents

**Goal:** build a minimal agent *from scratch* on top of the raw Anthropic
Messages API — no LangChain, no LangGraph — so tomorrow's frameworks feel
like conveniences, not magic.

This notebook is self-contained and reproducible. It imports `agent.py`
(the core loop + tool definitions) and `demos.py` (scripted test scenarios)
from this same folder. All outputs below were actually executed and saved —
nothing here is hand-typed console text.

**Two run modes, one identical loop:**
- If `ANTHROPIC_API_KEY` is set, the real Claude model can be used (see
  `live_demo.py` for a ready-to-run live version).
- Otherwise — and by default in this notebook, so it's fully reproducible
  without secrets — a `MockAnthropicClient` scripts realistic model
  behavior through the exact same `run_agent()` function. The loop code
  under test is 100% identical in both cases.
""")

# =============================================================================
md(r"""
## Task 1 — Agent Concepts & Mental Model

**Chatbot** — a single request/response turn (or a chat history replayed
each turn) with no ability to take actions in the world. It only ever
produces text back to the human.

**Workflow** — a *fixed*, human-designed sequence of steps (possibly
including LLM calls) wired together in code, e.g. "call the LLM to
summarize -> always call the translation API -> always email the result."
The control flow is decided in advance by the developer, not by the model.

**Agent** — the *model itself* decides, at each step, what to do next: which
tool (if any) to call, with what arguments, and when it has enough
information to stop and answer. The control flow lives inside the loop,
driven by the model's own reasoning, not hardcoded by the developer.

**What makes something "agentic"?**
- **Autonomy** — the system chooses its own next action rather than
  following a script.
- **Tool use** — it can act on the world (call functions/APIs), not just
  emit text.
- **Multi-step planning** — it can decompose a task into sub-steps across
  multiple turns.
- **Self-correction** — it can notice a tool failed or an answer was wrong,
  and adjust (retry, ask for clarification, try a different approach).

A system needs autonomy + the ability to observe results and change course;
a single LLM call that always does the same fixed thing is not "agentic"
even if it's smart.

### The ReAct pattern (Reason → Act → Observe → repeat)

```
 ┌──────────────────────────────────────────────────────────┐
 │                         while not done:                   │
 │                                                            │
 │   ┌─────────┐      ┌────────┐      ┌───────────┐          │
 │   │ REASON  │ ───▶ │  ACT   │ ───▶ │  OBSERVE   │──┐       │
 │   │ (model  │      │ (call  │      │ (tool      │  │       │
 │   │ decides │      │  tool) │      │  result    │  │       │
 │   │ next    │      │        │      │  fed back  │  │       │
 │   │ step)   │      │        │      │  in)       │  │       │
 │   └─────────┘      └────────┘      └───────────┘  │       │
 │        ▲                                            │       │
 │        └────────────────────────────────────────────┘       │
 │                                                            │
 │   if model returns plain text with no tool call -> DONE    │
 └──────────────────────────────────────────────────────────┘
```

Pseudocode:

```python
messages = [user_message]
while iterations < max_iterations:
    response = model(messages, tools)          # REASON
    if response.has_tool_call:
        result = execute(response.tool_call)   # ACT
        messages.append(tool_result(result))   # OBSERVE
    else:
        return response.text                   # DONE
raise StoppedByGuardrail("max_iterations reached")
```

### When is an agent overkill?

If a task has a **known, fixed sequence of steps** and doesn't need the
model to decide *what* to do next (only *how* to phrase something), a
single prompt or a plain script is faster, cheaper, and far more
predictable than an agent loop. For example, "translate this paragraph" or
"reformat this CSV" needs zero tool-choice reasoning — wrapping it in a
ReAct loop just adds latency, cost, and a new class of failure modes
(looping, wrong tool choice) for no benefit. Reach for an agent only when
the number and order of steps genuinely depends on information the model
discovers *during* the task.
""")

# =============================================================================
md(r"""
## Task 2 — Tool Calling Fundamentals

Below are the tool JSON schemas from `agent.py`. Each tool has a `name`,
a `description`, and an `input_schema` (JSON Schema).

**Why descriptions matter so much:** the model has *no other information*
about what a tool does besides its name, description, and schema. A vague
description like `"gets weather"` gives the model no signal about *when*
to call it, what format the input should be in, or what happens on
failure — which directly causes wrong-tool selection, malformed arguments,
and hallucinated capabilities. A good description states: (1) what it
does, (2) when to use it vs. not, (3) the expected input format /
constraints, and (4) what happens on edge cases (e.g. "returns an error if
the city isn't found — don't guess a number").
""")

code(r"""
import json
from agent import TOOLS

print(json.dumps(TOOLS, indent=2))
""")

# =============================================================================
md(r"""
### A single tool-use request/response, executed manually

This is the smallest possible demonstration of tool calling: send one
message, let the model choose a tool, manually run that tool, and manually
build the `tool_result` block — with no loop yet (the loop comes in Task 3).

We use a one-shot mock response here to show the exact shape of the
API objects; the same code works unmodified against a real
`anthropic.Anthropic()` client (see the `LIVE MODE` note above `agent.py`).
""")

code(r"""
from agent import MockAnthropicClient, _MockBlock, _MockMessage, make_id, execute_tool

# --- 1. Build a one-turn mock "model chose a tool" response -----------------
def turn1(messages):
    return _MockMessage(
        [_MockBlock(
            "tool_use",
            id=make_id("toolu", 1),
            name="get_weather",
            input={"city": "Lahore"},
        )],
        "tool_use",
    )

client = MockAnthropicClient([turn1])

messages = [{"role": "user", "content": "What's the weather in Lahore?"}]
response = client.messages.create(model="claude-sonnet-4-6", max_tokens=256,
                                   tools=[], messages=messages)

tool_call = response.content[0]
print("Model chose tool:", tool_call.name, "with input:", tool_call.input)

# --- 2. Manually execute the tool -------------------------------------------
result_text, is_error = execute_tool(tool_call.name, tool_call.input)
print("Tool result:", result_text, "| is_error:", is_error)

# --- 3. Manually build the tool_result block to send back -------------------
tool_result_block = {
    "type": "tool_result",
    "tool_use_id": tool_call.id,
    "content": result_text,
    "is_error": is_error,
}
print("\ntool_result block to send back to the model:")
print(json.dumps(tool_result_block, indent=2))
""")

# =============================================================================
md(r"""
## Task 3 — Minimal Agent Loop

`run_agent()` in `agent.py` implements the while-loop described in Task 1:
send message → check for `tool_use` → execute tool → append `tool_result` →
repeat until the model returns plain text, capped by `max_iterations`.

Below we run it on a task that genuinely requires **two sequential tool
calls**: *"Look up the weather in Lahore and Karachi and tell me which city
is warmer."* The model cannot answer this in one shot — it must call the
weather tool twice before it has enough information.
""")

code(r"""
from agent import run_agent
from demos import build_weather_comparison_client

result = run_agent(
    build_weather_comparison_client(),
    "Look up the weather in Lahore and Karachi and tell me which city is warmer.",
)
""")

code(r"""
print("Iterations used:", result["iterations_used"])
print("Stopped reason:", result["stopped_reason"])
print("\nFinal answer:\n", result["final_answer"])
""")

# =============================================================================
md(r"""
## Task 4 — Memory & State Handling

Two different kinds of "memory" are at play, and it's worth keeping them
conceptually separate even in a 100-line agent:

| | Conversation memory | Working memory |
|---|---|---|
| **What it is** | The literal message transcript (`messages` list) sent back to the model every single turn — user turns, assistant turns, tool_result turns. | A structured scratchpad *the agent code* maintains about what's happened so far (e.g. every tool called + its result). |
| **Who reads it** | The model — it's the model's only window into what happened earlier. | The developer / the agent's own logic — used for logging, debugging, deciding when to stop, or could be summarized and re-injected into the prompt. |
| **Grows how** | Every single turn, verbatim, including full tool outputs — can get large fast. | Only as much as you choose to extract/structure — can be much smaller and queryable. |
| **In our code** | `messages` inside `run_agent()`. | `working_memory = {"tool_calls": [...], "observations": [...]}` inside `run_agent()`. |

The logging below (`[model reasoning/text]`, `[ACT]`, `[OBSERVE]`) prints
every reasoning step, tool call, and observation as it happens — this is
the debugging habit worth carrying into every framework from tomorrow
onward, because frameworks often hide this trace behind an abstraction
unless you explicitly turn on verbose/callback logging.
""")

code(r"""
import json
print("Working memory captured during the run above:")
print(json.dumps(result["working_memory"], indent=2))

print("\nConversation memory length (number of turns sent to the model):",
      len(result["messages"]))
""")

# =============================================================================
md(r"""
## Task 5 — Failure Modes & Guardrails

We deliberately broke the agent five different ways using scripted mock
responses (see `demos.py`), so each failure is reproducible on demand
rather than something we happened to observe once.
""")

code(r"""
from demos import (
    build_infinite_loop_client,
    build_hallucinated_tool_client,
    build_wrong_arguments_client,
    build_tool_error_client,
    build_ambiguous_request_client,
)

print("##### 5.1 Infinite-loop tendency (capped by max_iterations) #####")
r1 = run_agent(build_infinite_loop_client(), "Keep telling me the Lahore weather over and over.", max_iterations=4)
""")

code(r"""
print("##### 5.2 Hallucinated tool call (model invents 'get_stock_price') #####")
r2 = run_agent(build_hallucinated_tool_client(), "What is Apple's current stock price?")
""")

code(r"""
print("##### 5.3 Wrong tool arguments (words instead of digits) #####")
r3 = run_agent(build_wrong_arguments_client(), "What's twenty three times four?")
""")

code(r"""
print("##### 5.4 Tool returns a real error (unsupported city) #####")
r4 = run_agent(build_tool_error_client(), "What's the weather in Atlantis?")
""")

code(r"""
print("##### 5.5 Ambiguous request (no city given, no tool call even attempted) #####")
r5 = run_agent(build_ambiguous_request_client(), "What's the weather?")
""")

# =============================================================================
md(r"""
### Failure modes observed → mitigations

| # | Failure mode | What we saw | Mitigation |
|---|---|---|---|
| 1 | **Infinite / repetitive tool-calling loop** | Model kept calling `get_weather("Lahore")` with no new information, never reaching a final answer. | Hard `max_iterations` cap in `run_agent()`; return a "best-effort / stopped" result instead of hanging forever. |
| 2 | **Hallucinated tool call** | Model called `get_stock_price`, a tool that was never registered. | `execute_tool()` checks the tool name against `TOOL_IMPLEMENTATIONS` and returns a structured `is_error=True` result instead of crashing, so the model sees the failure and can say "I can't do that." |
| 3 | **Wrong / malformed tool arguments** | Model passed `"twenty three times four"` instead of a numeric expression. | Input validation inside the tool itself (`tool_calculator` whitelists characters and raises `ValueError`); the error is surfaced as a `tool_result` so the model can retry with corrected arguments — which it did, successfully, on the next turn. |
| 4 | **Tool executes but returns a domain error** | `get_weather("Atlantis")` — a well-formed call to a real tool, but the underlying data doesn't exist. | Tool raises a clear, descriptive exception; `execute_tool()` catches *all* exceptions generically and reports them as `is_error=True` rather than letting the whole process crash. |
| 5 | **Silent / swallowed errors** | A tempting anti-pattern: catching an exception and returning `""` or `None` looks "safe" but the model has no idea anything went wrong, and may confidently make something up. | Never return an empty/blank string on failure — always return a descriptive error message *and* set `is_error=True`, so the model treats it as a real observation, not an empty success. |
| 6 | **Ambiguous user request** | "What's the weather?" with no city — nothing to even call a tool with. | Give tool descriptions that state their required inputs clearly, and instruct the model (via `system_prompt`) to ask a clarifying question rather than guessing a default city. |

### Why do frameworks like LangChain / LangGraph / CrewAI exist?

Having now built this by hand, the value of a framework becomes obvious —
and so do its costs. The 150-ish lines above already needed careful
handling of message formatting, tool-result schemas, error propagation, and
a loop guard; a real production agent also wants things like: automatic
context-window trimming/summarization of long conversation memory, retries
with backoff on API errors, streaming, parallel tool execution, structured
multi-agent handoffs, built-in tracing/observability, and a standard vocabulary
so teams don't reinvent this loop slightly differently every time. Frameworks
package those recurring, tedious, easy-to-get-subtly-wrong pieces into
reusable primitives. The tradeoff is that they also hide the loop — which is
exactly why doing it in raw Python first matters: tomorrow, when a framework's
`AgentExecutor` or graph node behaves unexpectedly, we'll know precisely what
it's doing underneath, because we built the "underneath" ourselves today.
""")

nb['cells'] = cells

with open('/home/claude/week2day1_agent/agent_foundations.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook written.")
