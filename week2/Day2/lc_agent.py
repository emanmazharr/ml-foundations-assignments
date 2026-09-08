"""
Week 2 / Day 2 — LangChain agent, memory, and structured output.

Uses langchain-google-genai (Gemini) as the LLM backend because it has a
free tier with no billing card required, so it's the fastest path to a
genuinely working live run. Swapping to langchain-anthropic's ChatAnthropic
or any other langchain chat model is a one-line change — everything below
(tools, agent, memory, structured output) is model-agnostic LangChain code.

Run `pip install -r requirements.txt` (see below) and set
GOOGLE_API_KEY before importing this module for live use.
"""

import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from pydantic import BaseModel, Field

from lc_tools import TOOLS


def get_llm(model_name="gemini-2.0-flash", temperature=0):
    """Return a configured chat model. Requires GOOGLE_API_KEY to be set.

    NOTE: switched default from 'gemini-3.6-flash' to 'gemini-2.0-flash'.
    Free-tier quota (20 requests/day) is tracked per (project, model) pair,
    so a model you haven't called yet today has its own fresh 20-request
    allowance even on the same API key/project that just hit its limit on
    a different model.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/app/apikey and set it with "
            "os.environ['GOOGLE_API_KEY'] = '...' before calling get_llm()."
        )
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


# ---------------------------------------------------------------------------
# TASK 1: a basic LCEL prompt -> response pipeline
# ---------------------------------------------------------------------------
def build_basic_chain(llm):
    """
    The simplest possible LCEL pipeline: prompt | llm | output_parser.

    The `|` operator overloads Python's bitwise-or to mean "pipe the output
    of the thing on the left into the thing on the right." Every piece
    (prompt template, chat model, output parser) implements the same
    `Runnable` interface (`.invoke()`, `.stream()`, `.batch()`), so `|`
    just chains their `.invoke()` calls together into one new Runnable —
    it's function composition, not magic: `chain.invoke(x)` is equivalent
    to `output_parser.invoke(llm.invoke(prompt.invoke(x)))`.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", "You are a concise assistant."), ("human", "{question}")]
    )
    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# TASK 3: tool-calling agent + AgentExecutor
# ---------------------------------------------------------------------------
def build_agent_executor(llm, verbose=True):
    """
    LangChain equivalents of Day 1's raw-Python pieces:

      Day 1 (raw)                          LangChain
      ------------------------------------ -----------------------------------
      anthropic.Anthropic() client       -> `llm` (a `BaseChatModel`, e.g.
                                             ChatGoogleGenerativeAI)
      TOOLS list of JSON schemas +
      TOOL_IMPLEMENTATIONS dict           -> `@tool`-decorated Python functions
                                             (the docstring IS the schema)
      run_agent() while-loop              -> `AgentExecutor` (runs the
      (reason -> act -> observe -> repeat)   reason/act/observe loop for you)
      messages list (conversation memory) -> chat history object
                                             (ConversationBufferMemory /
                                             RunnableWithMessageHistory)
      working_memory scratchpad + log()   -> AgentExecutor's `verbose=True`
                                             trace + LangChain callbacks
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful assistant with access to a calculator, "
                "a weather lookup tool, and a product price lookup tool. "
                "Always use a tool instead of guessing a number. If a tool "
                "errors or a request is ambiguous, say so plainly.",
            ),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    # handle_parsing_errors + tools' own try/except-free design (they raise
    # real exceptions) together give the Task 5 "graceful recovery"
    # behaviour: AgentExecutor catches a tool exception, turns it into a
    # ToolException/error string, and feeds it back to the model as an
    # observation instead of crashing the whole run.
    return AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
    )


# ---------------------------------------------------------------------------
# TASK 4: memory across turns
# ---------------------------------------------------------------------------
_SESSION_STORE: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = InMemoryChatMessageHistory()
    return _SESSION_STORE[session_id]


def build_agent_with_memory(agent_executor):
    """
    Wrap the AgentExecutor with RunnableWithMessageHistory (the modern
    replacement for ConversationBufferMemory). This is what lets a 3-turn
    conversation like:

        "Find the price of the UltraBook Pro"
        "Now compare it to the BudgetBook Lite"
        "Which one should I recommend to a budget-conscious client?"

    work correctly — turn 2 and 3 only make sense if the model can see
    turn 1's tool result in its chat history, which RunnableWithMessageHistory
    injects automatically into the `chat_history` placeholder above.
    """
    return RunnableWithMessageHistory(
        agent_executor,
        _get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )


# ---------------------------------------------------------------------------
# TASK 5: structured output
# ---------------------------------------------------------------------------
class Recommendation(BaseModel):
    """Structured final answer for a product recommendation."""

    recommended_product: str = Field(description="Name of the recommended product")
    price_usd: float = Field(description="Price of the recommended product in USD")
    price_difference_usd: float = Field(
        description="Absolute price difference vs. the other product compared"
    )
    reasoning: str = Field(description="1-2 sentence justification for the recommendation")


def build_structured_formatter(llm):
    """
    A small second chain that takes the agent's free-text final answer and
    forces it into the `Recommendation` Pydantic schema, using LangChain's
    `with_structured_output`. Keeping this as a separate formatting step
    (rather than forcing the tool-calling agent itself into structured
    output) avoids a known rough edge: many models can't reliably do
    tool-calling AND schema-constrained output in the same call.
    """
    structured_llm = llm.with_structured_output(Recommendation)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Extract a structured recommendation from the assistant's answer below.",
            ),
            ("human", "{agent_answer}"),
        ]
    )
    return prompt | structured_llm
