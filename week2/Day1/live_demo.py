"""
Run the exact same agent loop against the REAL Anthropic API instead of the
mock client, so Claude itself decides which tools to call and in what order.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -c "import anthropic; print(anthropic.__version__)"
"""

import sys
from agent import run_agent, get_client

SYSTEM_PROMPT = (
    "You are a careful assistant with access to a calculator, a weather "
    "lookup tool, and a text file reader. Use tools whenever a question "
    "depends on a calculation, current weather, or file contents instead "
    "of guessing. If a tool errors or a request is ambiguous, say so "
    "plainly instead of making up an answer."
)

if __name__ == "__main__":
    client = get_client()
    if client is None:
        print(
            "No ANTHROPIC_API_KEY found in the environment.\n"
            "Set it first, e.g.:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Falling back is not done here on purpose — see demos.py for "
            "the mock-mode version of this same task."
        )
        sys.exit(1)

    task = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Look up the weather in Lahore and Karachi and tell me which city is warmer, "
        "then tell me the temperature difference using the calculator."
    )

    result = run_agent(client, task, system_prompt=SYSTEM_PROMPT, max_iterations=8)
    print("\n\nSTOPPED REASON:", result["stopped_reason"])
    print("ITERATIONS USED:", result["iterations_used"])
