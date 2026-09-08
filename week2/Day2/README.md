# Week 2 · Day 2 — LangChain: Tools, Chains, Memory & Your First Framework Agent

Rebuilds Day 1's raw-Python agent using LangChain, with 3 tools (one
reading from a real external JSON data source), a tool-calling agent with
a verbose reasoning trace, cross-turn memory, and structured output.

## Files

| File | Purpose |
|---|---|
| `langchain_agent.ipynb` | Main deliverable — all 5 tasks, with explanations. **Must be run** (see below) to produce real output before submission. |
| `lc_tools.py` | 3 tools defined with `@tool`: `calculator`, `get_weather`, `get_product_price` (reads `products.json`). |
| `lc_agent.py` | LCEL chain, `create_tool_calling_agent` + `AgentExecutor`, `RunnableWithMessageHistory` memory, Pydantic structured output. |
| `products.json` | Local JSON "database" of laptop prices — the external data source for `get_product_price`. |
| `requirements.txt` | `pip install -r requirements.txt` |
| `WRITEUP_day2.md` | 1-page write-up: concept mapping, LCEL explanation, annotated trace, memory, structured output, and raw-vs-framework comparison. |

## How to run (Google Colab — fastest, ~2 minutes setup)

1. Go to https://colab.research.google.com, upload `langchain_agent.ipynb`.
2. In the Colab file sidebar (folder icon on the left), upload `lc_tools.py`,
   `lc_agent.py`, and `products.json` into the same session.
3. Get a **free** Gemini API key at https://aistudio.google.com/app/apikey
   (no credit card / billing setup required — unlike some other providers).
4. Run the notebook top to bottom: **Runtime → Run all**. It'll prompt you
   to paste your API key in the second code cell.
5. Once it finishes, download the notebook again (**File → Download → .ipynb**)
   so the saved copy includes the real executed output, then submit that
   version alongside the other files.

## Why Gemini instead of Anthropic here

`langchain-google-genai` was used because Google AI Studio issues a free
API key instantly with no billing setup — the fastest way to get a fully
working live run under time pressure. Every tool, chain, agent, memory,
and structured-output piece in `lc_agent.py` is plain LangChain code and
is model-agnostic; swapping in `langchain-anthropic`'s `ChatAnthropic` (or
any other supported chat model) only requires changing `get_llm()`.
