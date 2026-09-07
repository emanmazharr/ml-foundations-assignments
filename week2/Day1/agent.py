"""
Week 2 / Day 1 — Agent Foundations
==================================
A minimal, dependency-free (besides the `anthropic` SDK) ReAct-style agent
built directly on top of the Anthropic Messages API — no LangChain, no
LangGraph. The goal is to see exactly what a "framework" is hiding from you.

This module can run in two modes, selected automatically:

1. LIVE MODE  — if the environment variable ANTHROPIC_API_KEY is set, we use
   the real `anthropic.Anthropic()` client and Claude actually decides which
   tools to call.

2. MOCK MODE  — if no API key is present, we use `MockAnthropicClient`, a
   tiny scripted stand-in that implements the SAME interface
   (`.messages.create(...)` -> object with `.content` and `.stop_reason`).
   This lets the ENTIRE agent loop, logging, memory handling, and failure
   scenarios run end-to-end and be committed to the repo with real,
   reproducible console output — without requiring secrets in CI or in a
   grader's environment.

Both modes flow through the exact same `run_agent()` loop below. That is the
whole point of the exercise: the loop doesn't care who is on the other end
of `.messages.create()`.
"""

import os
import json
import traceback

# ---------------------------------------------------------------------------
# TASK 2: TOOL DEFINITIONS (JSON schemas)
# ---------------------------------------------------------------------------
# Anthropic's tool-use feature expects a list of tool specs, each with:
#   - name:        the identifier the model will use to call it
#   - description: PLAIN ENGLISH explanation of what it does and when to use
#                  it — this is the single biggest lever for reliable tool
#                  calling. The model has no other information about your
#                  tool except this description + the schema, so vague or
#                  missing descriptions are the #1 cause of wrong-tool /
#                  wrong-argument calls.
#   - input_schema: a JSON Schema describing the expected arguments.

TOOLS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate a basic arithmetic expression and return the numeric "
            "result. Use this any time the user asks for a calculation, "
            "comparison of numbers, unit conversion arithmetic, or any math "
            "you should not attempt to do 'in your head'. Only supports "
            "+, -, *, /, %, ** and parentheses. Do NOT pass words, only a "
            "valid arithmetic expression, e.g. '23 * (4 + 1)'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid arithmetic expression, e.g. '18 - 3.5 * 2'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Look up the CURRENT weather for a named city. Returns the "
            "temperature in Celsius and a short condition string (e.g. "
            "'sunny', 'rainy'). Use this whenever the user asks about "
            "weather, temperature, or wants to compare conditions between "
            "cities. This is a stub/demo tool backed by a small fixed "
            "lookup table, not a live weather feed — if the city is not in "
            "the table it will return an error, which you should surface "
            "to the user rather than guessing a number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Lahore' or 'Karachi'",
                }
            },
            "required": ["city"],
        },
    },
    {
        "name": "read_text_file",
        "description": (
            "Read and return the full text contents of a local .txt file "
            "given its filename. Use this when the user asks about the "
            "contents of a specific file that has already been mentioned "
            "or uploaded. Only .txt files in the current working directory "
            "are supported; anything else returns an error."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the .txt file, e.g. 'notes.txt'",
                }
            },
            "required": ["filename"],
        },
    },
]

# ---------------------------------------------------------------------------
# TOOL IMPLEMENTATIONS
# ---------------------------------------------------------------------------
# A tiny fixed lookup table standing in for a real weather API call.
_FAKE_WEATHER_DB = {
    "lahore": {"temp_c": 34, "condition": "hazy sunshine"},
    "karachi": {"temp_c": 31, "condition": "humid, partly cloudy"},
    "islamabad": {"temp_c": 27, "condition": "light rain"},
    "london": {"temp_c": 18, "condition": "overcast"},
    "dubai": {"temp_c": 41, "condition": "clear and very hot"},
}


def tool_calculator(expression: str) -> str:
    """Safely evaluate a restricted arithmetic expression."""
    allowed_chars = set("0123456789.+-*/()% \t")
    if not set(expression) <= allowed_chars:
        raise ValueError(
            f"Expression '{expression}' contains characters that are not "
            "plain arithmetic. Refusing to evaluate for safety."
        )
    # eval is safe here ONLY because we whitelisted the character set above
    # (no names, no attribute access, no builtins can be referenced).
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)


def tool_get_weather(city: str) -> str:
    key = city.strip().lower()
    if key not in _FAKE_WEATHER_DB:
        # Deliberately raise so the agent has to handle a real tool error
        # (see Task 5 — this is one of our designed failure modes).
        raise KeyError(
            f"No weather data available for '{city}'. Known cities: "
            f"{', '.join(c.title() for c in _FAKE_WEATHER_DB)}"
        )
    data = _FAKE_WEATHER_DB[key]
    return json.dumps({"city": city, **data})


def tool_read_text_file(filename: str) -> str:
    if not filename.endswith(".txt"):
        raise ValueError("Only .txt files are supported by this stub tool.")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' was not found.")
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


TOOL_IMPLEMENTATIONS = {
    "calculator": lambda inp: tool_calculator(inp["expression"]),
    "get_weather": lambda inp: tool_get_weather(inp["city"]),
    "read_text_file": lambda inp: tool_read_text_file(inp["filename"]),
}


def execute_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """
    Execute a tool by name. Returns (result_text, is_error).
    Never raises — all failures are caught and turned into a tool_result
    with is_error=True, exactly like the real Anthropic tool-use contract
    expects, so the model can see the failure and recover.
    """
    if name not in TOOL_IMPLEMENTATIONS:
        # This is the "hallucinated tool call" failure mode (Task 5): the
        # model asked for a tool that was never defined/registered.
        return (
            f"Error: tool '{name}' is not implemented/registered. "
            f"Available tools: {list(TOOL_IMPLEMENTATIONS)}",
            True,
        )
    try:
        result = TOOL_IMPLEMENTATIONS[name](tool_input)
        return (result, False)
    except Exception as e:  # noqa: BLE001 - intentionally broad, see Task 5
        return (f"Error executing tool '{name}': {e}", True)


# ---------------------------------------------------------------------------
# MOCK CLIENT — implements the same shape as anthropic.Anthropic().messages
# ---------------------------------------------------------------------------
class _MockBlock:
    """Mimics a content block returned by the real SDK (text or tool_use)."""

    def __init__(self, type_, **kwargs):
        self.type = type_
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockMessage:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class MockAnthropicClient:
    """
    A deterministic, scripted stand-in for the real Anthropic client.

    `script` is a list of "turns". Each turn is a function that receives the
    current message history and returns a _MockMessage. This lets us encode
    exactly the multi-step behavior we want to demonstrate (2+ tool calls,
    an infinite-loop tendency, a hallucinated tool, etc.) without needing
    network access or an API key, while exercising the *real* run_agent()
    loop below unmodified.
    """

    def __init__(self, script):
        self._script = script
        self._turn = 0
        self.messages = self  # so `client.messages.create(...)` works

    def create(self, model, max_tokens, tools, messages, system=None):
        if self._turn >= len(self._script):
            # Fallback: just end the conversation politely.
            return _MockMessage(
                [_MockBlock("text", text="(mock client ran out of script)")],
                "end_turn",
            )
        turn_fn = self._script[self._turn]
        self._turn += 1
        return turn_fn(messages)


def make_id(prefix, n):
    return f"{prefix}_{n:03d}"


# ---------------------------------------------------------------------------
# TASK 3: THE AGENT LOOP (ReAct: Reason -> Act -> Observe -> repeat)
# ---------------------------------------------------------------------------
def run_agent(
    client,
    user_message: str,
    tools=TOOLS,
    model="claude-sonnet-4-6",
    max_iterations=6,
    system_prompt=None,
    verbose=True,
):
    """
    Runs a minimal ReAct loop:

        while not done:
            response = ask the model (Reason)
            if response wants a tool call:
                execute the tool (Act)
                feed the result back in as an observation (Observe)
            else:
                return the model's final text answer

    Two kinds of memory are tracked separately here (see Task 4):
      - `messages` is CONVERSATION MEMORY: the literal transcript sent back
        to the model every turn (user, assistant, tool_result...).
      - `working_memory` is WORKING MEMORY / scratchpad: structured facts
        the AGENT (not the model) extracts and accumulates as it goes,
        e.g. every city+temperature it has looked up so far. The model
        never sees `working_memory` directly — it's for our own bookkeeping,
        debugging, and could be re-injected into the prompt if needed.
    """
    messages = [{"role": "user", "content": user_message}]
    working_memory = {"tool_calls": [], "observations": []}

    def log(msg):
        if verbose:
            print(msg)

    log(f"\n{'='*70}\nUSER: {user_message}\n{'='*70}")

    for step in range(1, max_iterations + 1):
        log(f"\n--- Iteration {step}/{max_iterations} (REASON) ---")

        kwargs = dict(
            model=model,
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        # Log any reasoning/plain-text the model produced this turn.
        for block in response.content:
            if block.type == "text" and block.text.strip():
                log(f"[model reasoning/text] {block.text.strip()}")

        # Append the assistant's turn to conversation memory verbatim.
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No tool requested -> model is done. Return final answer.
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            ).strip()
            log(f"\n--- FINAL ANSWER (after {step} iteration(s)) ---\n{final_text}")
            return {
                "final_answer": final_text,
                "iterations_used": step,
                "working_memory": working_memory,
                "messages": messages,
                "stopped_reason": "final_answer",
            }

        # --- ACT: execute every requested tool call ---
        tool_results = []
        for block in tool_use_blocks:
            log(f"[ACT] calling tool '{block.name}' with input {block.input}")
            result_text, is_error = execute_tool(block.name, block.input)

            # --- OBSERVE: log + store the outcome ---
            log(f"[OBSERVE] {'ERROR' if is_error else 'result'}: {result_text}")

            working_memory["tool_calls"].append(
                {"tool": block.name, "input": block.input}
            )
            working_memory["observations"].append(
                {"tool": block.name, "result": result_text, "is_error": is_error}
            )

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": is_error,
                }
            )

        # Feed observations back in as the next "user" turn (this is how the
        # Anthropic API expects tool results to be returned).
        messages.append({"role": "user", "content": tool_results})

    # --- Safeguard: max_iterations reached without a final answer ---
    log(
        f"\n--- STOPPED: max_iterations ({max_iterations}) reached without "
        "a final answer. Returning best-effort summary. ---"
    )
    return {
        "final_answer": None,
        "iterations_used": max_iterations,
        "working_memory": working_memory,
        "messages": messages,
        "stopped_reason": "max_iterations_reached",
    }


def get_client():
    """Return a real Anthropic client if a key is configured, else None."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=api_key)
