# Week 3 Day 3 — Domain-Scoped AFL Chat Agent (Gemini Version)

This submission uses the supplied real AFL datasets and a LangChain tool-calling agent powered by Google Gemini.

## Run
1. Install dependencies:
   `pip install -r requirements.txt`
2. Set your Gemini API key as an environment variable. **Do not put the key inside the submitted code.**
   - Windows PowerShell: `$env:GEMINI_API_KEY="YOUR_KEY"`
   - Windows CMD: `set GEMINI_API_KEY=YOUR_KEY`
3. Run the agent:
   `python afl_chat_agent.py`
4. Run the 18-prompt guardrail evaluation:
   `python evaluate_guardrails.py`

## Model
Default model: `gemini-3.6-flash`. You can change it without editing code:
`$env:GEMINI_MODEL="gemini-3.8-flash"`

The Gemini API supports tool/function calling, and the agent uses LangChain tools to retrieve exact values from the local AFL datasets before answering statistical questions. Google recommends environment variables such as `GEMINI_API_KEY` for API authentication.

## Tasks covered
- Task 1: AFL-only system prompt, refusal behavior, adversarial guardrail tests
- Task 2: Structured pandas retrieval over real AFL data
- Task 3: LangChain Gemini tool-calling + grounding through tool outputs
- Task 4: Conversation history for multi-turn follow-ups
- Task 5: 18-prompt guardrail evaluation and report
