# Week 3 Day 3 — Domain-Scoped AFL Chat Agent Evaluation

## Task 1 — Scope and refusal behavior
The system prompt limits the assistant to AFL teams, players, matches, statistics, history, rules and competitions. It explicitly excludes other sports, general trivia, unrelated coding, politics, entertainment and general chit-chat. Off-topic requests receive a short refusal followed by an AFL redirect.

Three refusal examples:
1. “I can help with AFL teams, players, matches, stats, history and rules. Please ask me an AFL-related question.”
2. “That is outside my AFL scope. I can help with an AFL player, team, match or statistic instead.”
3. “I only answer AFL-related questions. If you want, ask me about an AFL match, player or rule.”

The automated evaluation contains 18 prompts covering normal AFL questions, topic drift, indirect off-topic requests and instruction-injection wording.

## Task 2 — Retrieval design
Structured retrieval is used for exact numerical facts because the supplied data is tabular. Pandas lookups are used for:
- player season totals/averages;
- player match-by-match recent rounds;
- player recent-vs-career averages;
- team-vs-team historical records;
- player identity resolution.

A vector store is not added because the supplied files are structured CSV tables rather than unstructured match reports/news. Adding semantic retrieval to numeric tables would make exact-stat grounding less reliable and is unnecessary for this dataset.

## Task 3 — LangChain tool calling and grounding
The agent is built with LangChain and `ChatAnthropic`, using `create_tool_calling_agent` and `AgentExecutor`. Each retrieval function is registered as a LangChain tool with a clear docstring.

Grounding verification: `AgentExecutor` returns intermediate tool observations. `AFLChatAgent.last_tool_results` stores those observations. The system prompt requires every numerical/statistical answer to be based on a tool result and forbids guessing when the dataset has no value. Thus the final answer can be checked against the recorded tool output.

## Task 4 — Memory and multi-turn behavior
`AFLChatAgent.history` stores prior user and assistant messages and passes them back to the agent on every turn. This supports follow-ups such as:
1. identify a player;
2. ask for a season statistic;
3. ask “what about the round before that?”;
4. ask “how does that compare with his career average?”

The new factual value is still retrieved from the dataset rather than inferred from memory.

## Task 5 — Guardrail evaluation
The test set contains 18 prompts. It mixes legitimate AFL questions, clearly off-topic requests, ambiguous AFL-adjacent wording and adversarial instruction attempts. Results are written to `guardrail_evaluation.json` when the evaluation script is run.

### Failure patterns and fixes
- **Topic drift:** unrelated sports can appear in natural language. **Fix:** explicit out-of-scope terms plus an AFL relevance check before agent execution.
- **Instruction injection:** prompts such as “ignore your instructions” can attempt to widen scope. **Fix:** keep the scope in the system message and reject adversarial off-topic requests before tool use.
- **Hallucinated numbers:** a language model may know plausible AFL statistics from training. **Fix:** require the appropriate structured tool for numerical facts and retain intermediate tool results for verification.
- **Missing/ambiguous player names:** a name can map to multiple dataset records. **Fix:** player resolution returns an ambiguity message rather than silently choosing a record.

## Dataset
All retrieval is based on the supplied AFL CSV datasets: player information, player seasonal statistics, player round-by-round statistics, and team match results.
