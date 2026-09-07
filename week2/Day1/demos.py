"""
Demo scenarios for the Week 2 / Day 1 agent.

Each function below builds a deterministic MockAnthropicClient "script" that
stands in for what Claude would plausibly do in that situation, then runs it
through the REAL, unmodified `run_agent()` loop from agent.py. This proves
the loop's control flow (reason -> act -> observe -> repeat, memory
handling, logging, max_iterations safeguard, error handling) all work
correctly, independent of whether a live API key is available.

To run the SAME scenarios against the real Claude model instead, just swap
`client = build_mock_..._client()` for `client = agent.get_client()`
wherever it appears below (requires ANTHROPIC_API_KEY to be set).
"""

from agent import (
    run_agent,
    MockAnthropicClient,
    _MockBlock,
    _MockMessage,
    make_id,
)


# ---------------------------------------------------------------------------
# TASK 3: Happy path — a task that genuinely requires 2+ tool calls
# ---------------------------------------------------------------------------
def build_weather_comparison_client():
    """
    Scripted behavior for: "Look up the weather in Lahore and Karachi and
    tell me which city is warmer."
    Turn 1: model asks for Lahore's weather.
    Turn 2: model asks for Karachi's weather.
    Turn 3: model has both observations in its context and answers.
    """

    def turn1(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "text",
                    text="I need the current temperature for both cities before I can compare them. Starting with Lahore.",
                ),
                _MockBlock(
                    "tool_use",
                    id=make_id("toolu", 1),
                    name="get_weather",
                    input={"city": "Lahore"},
                ),
            ],
            "tool_use",
        )

    def turn2(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "text", text="Got Lahore. Now checking Karachi."
                ),
                _MockBlock(
                    "tool_use",
                    id=make_id("toolu", 2),
                    name="get_weather",
                    input={"city": "Karachi"},
                ),
            ],
            "tool_use",
        )

    def turn3(messages):
        # In a live run the model would read the actual tool_result content
        # from `messages` to produce this; here we hardcode the answer that
        # matches our fixed weather lookup table (Lahore 34C > Karachi 31C).
        return _MockMessage(
            [
                _MockBlock(
                    "text",
                    text=(
                        "Lahore is currently 34°C (hazy sunshine) and Karachi "
                        "is 31°C (humid, partly cloudy). Lahore is the warmer "
                        "of the two right now, by 3°C."
                    ),
                )
            ],
            "end_turn",
        )

    return MockAnthropicClient([turn1, turn2, turn3])


# ---------------------------------------------------------------------------
# TASK 5 / Failure mode 1: infinite-loop tendency
# ---------------------------------------------------------------------------
def build_infinite_loop_client(n_turns=10):
    """
    Model keeps re-calling the same tool with the same (already-answered)
    question and never converges to a final answer. Without max_iterations
    this would run forever. We script `n_turns` repeated tool_use turns to
    show the safeguard kicking in.
    """

    def make_turn(i):
        def turn(messages):
            return _MockMessage(
                [
                    _MockBlock(
                        "text", text=f"Let me check the weather again ({i})."
                    ),
                    _MockBlock(
                        "tool_use",
                        id=make_id("toolu", i),
                        name="get_weather",
                        input={"city": "Lahore"},
                    ),
                ],
                "tool_use",
            )

        return turn

    return MockAnthropicClient([make_turn(i) for i in range(1, n_turns + 1)])


# ---------------------------------------------------------------------------
# TASK 5 / Failure mode 2: hallucinated tool call
# ---------------------------------------------------------------------------
def build_hallucinated_tool_client():
    """Model invents a tool ('get_stock_price') that was never registered."""

    def turn1(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "tool_use",
                    id=make_id("toolu", 1),
                    name="get_stock_price",  # never defined in TOOLS!
                    input={"ticker": "AAPL"},
                )
            ],
            "tool_use",
        )

    def turn2(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "text",
                    text=(
                        "I don't actually have a way to look up stock prices "
                        "with the tools available to me, so I can't answer "
                        "that reliably."
                    ),
                )
            ],
            "end_turn",
        )

    return MockAnthropicClient([turn1, turn2])


# ---------------------------------------------------------------------------
# TASK 5 / Failure mode 3: wrong tool arguments
# ---------------------------------------------------------------------------
def build_wrong_arguments_client():
    """Model passes a natural-language string instead of a math expression."""

    def turn1(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "tool_use",
                    id=make_id("toolu", 1),
                    name="calculator",
                    input={"expression": "twenty three times four"},
                )
            ],
            "tool_use",
        )

    def turn2(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "text",
                    text=(
                        "That expression wasn't valid arithmetic, let me "
                        "retry with digits instead."
                    ),
                ),
                _MockBlock(
                    "tool_use",
                    id=make_id("toolu", 2),
                    name="calculator",
                    input={"expression": "23 * 4"},
                ),
            ],
            "tool_use",
        )

    def turn3(messages):
        return _MockMessage(
            [_MockBlock("text", text="23 times 4 is 92.")], "end_turn"
        )

    return MockAnthropicClient([turn1, turn2, turn3])


# ---------------------------------------------------------------------------
# TASK 5 / Failure mode 4: tool returns a real error (unsupported input)
# ---------------------------------------------------------------------------
def build_tool_error_client():
    """Weather asked for a city that is not in our fake DB."""

    def turn1(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "tool_use",
                    id=make_id("toolu", 1),
                    name="get_weather",
                    input={"city": "Atlantis"},
                )
            ],
            "tool_use",
        )

    def turn2(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "text",
                    text=(
                        "I don't have weather data for Atlantis in my "
                        "lookup tool, and I won't guess a number — could "
                        "you confirm the city name, or try Lahore, "
                        "Karachi, Islamabad, London, or Dubai?"
                    ),
                )
            ],
            "end_turn",
        )

    return MockAnthropicClient([turn1, turn2])


# ---------------------------------------------------------------------------
# TASK 5 / Failure mode 5: ambiguous request (no tool needed to expose it)
# ---------------------------------------------------------------------------
def build_ambiguous_request_client():
    """User just says 'what's the weather?' with no city at all."""

    def turn1(messages):
        return _MockMessage(
            [
                _MockBlock(
                    "text",
                    text=(
                        "I'd be glad to check the weather, but I need to "
                        "know which city you mean first — the tool I have "
                        "requires a specific city name."
                    ),
                )
            ],
            "end_turn",
        )

    return MockAnthropicClient([turn1])


if __name__ == "__main__":
    print("\n\n########## TASK 3: MULTI-STEP HAPPY PATH ##########")
    run_agent(
        build_weather_comparison_client(),
        "Look up the weather in Lahore and Karachi and tell me which city is warmer.",
    )

    print("\n\n########## TASK 5.1: INFINITE-LOOP TENDENCY (max_iterations guard) ##########")
    run_agent(
        build_infinite_loop_client(),
        "Keep telling me the Lahore weather over and over.",
        max_iterations=4,
    )

    print("\n\n########## TASK 5.2: HALLUCINATED TOOL CALL ##########")
    run_agent(
        build_hallucinated_tool_client(),
        "What is Apple's current stock price?",
    )

    print("\n\n########## TASK 5.3: WRONG TOOL ARGUMENTS ##########")
    run_agent(
        build_wrong_arguments_client(),
        "What's twenty three times four?",
    )

    print("\n\n########## TASK 5.4: TOOL RETURNS AN ERROR ##########")
    run_agent(
        build_tool_error_client(),
        "What's the weather in Atlantis?",
    )

    print("\n\n########## TASK 5.5: AMBIGUOUS REQUEST ##########")
    run_agent(
        build_ambiguous_request_client(),
        "What's the weather?",
    )
