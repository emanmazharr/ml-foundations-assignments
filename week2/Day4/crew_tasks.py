"""
Week 2 / Day 4 — Task definitions.

Task 3 asks us to note a place where one task's output wasn't in the
format the next agent needed, and how the prompt/expected_output was
fixed. That happened here during design, not by accident at runtime:

  BEFORE (loose): the analyst's expected_output just said "list the
  products and their prices." In an early draft this let the model
  paraphrase/round numbers (e.g. "around $1,500" instead of "1499"),
  which then broke the strategist's price_value_calculator call in
  Task 2 — the strategist was handed an approximate string, not a clean
  number, and its division expression became invalid/inconsistent.

  AFTER (fixed, what's shipped below): task_analyze's description and
  expected_output were tightened to explicitly forbid rounding or any
  invented detail, and to require the EXACT catalog field names/values
  verbatim — because "list the products" left too much room for the
  model to paraphrase data that downstream code (the calculator tool)
  needed to parse as a literal number.
"""

from crewai import Task


def build_sequential_tasks(analyst, strategist, writer):
    task_analyze = Task(
        description=(
            "Call the list_products tool to get the full laptop catalog. "
            "Report EVERY product with its EXACT price_usd, category, and "
            "specs, copied verbatim from the tool's output. Do NOT round "
            "any number, do NOT paraphrase any spec, and do NOT add any "
            "opinion, comparison, or recommendation — facts only."
        ),
        expected_output=(
            "A bulleted list, one bullet per product, each showing the "
            "exact product name, price_usd (as a plain integer, no "
            "rounding or currency-word paraphrasing), category, and specs "
            "string exactly as returned by the tool."
        ),
        agent=analyst,
    )

    task_strategize = Task(
        description=(
            "Using ONLY the analyst's catalog facts above (do not invent "
            "any product or number not listed there), identify 2-3 "
            "distinct customer segments (e.g. budget-conscious students, "
            "portability-focused professionals, performance-focused power "
            "users). For each segment, recommend the best-fit laptop with "
            "one sentence of justification. Use the price_value_calculator "
            "tool to compute price_usd divided by the RAM in GB for at "
            "least two products, and cite that number to support your "
            "reasoning."
        ),
        expected_output=(
            "One short section per customer segment: segment name, "
            "recommended laptop, a one-sentence justification, and at "
            "least one computed price-per-GB-RAM value backing the choice."
        ),
        agent=strategist,
        context=[task_analyze],
    )

    task_write = Task(
        description=(
            "Using ONLY the strategist's segment recommendations above "
            "(do not introduce any new fact, product, or number), write a "
            "concise, polished executive summary under 200 words for a "
            "back-to-school marketing campaign. Open with a one-sentence "
            "overview, one short paragraph per segment/recommended "
            "laptop, and close with a single named 'hero product' pick "
            "for the campaign."
        ),
        expected_output=(
            "A polished executive summary under 200 words, stakeholder-"
            "ready, ending with one clearly labeled hero product pick."
        ),
        agent=writer,
        context=[task_analyze, task_strategize],
    )

    return [task_analyze, task_strategize, task_write]


def build_hierarchical_tasks():
    """
    Same three pieces of work, but with NO agent pre-assigned — in
    Process.hierarchical, the manager agent is responsible for deciding
    which of its crew members handles each task and for reviewing their
    output, so leaving `agent` unset here is what actually exercises real
    delegation rather than just running the same fixed assignment through
    a manager wrapper.
    """
    task_analyze = Task(
        description=(
            "Call the list_products tool to get the full laptop catalog. "
            "Report EVERY product with its EXACT price_usd, category, and "
            "specs, copied verbatim from the tool's output. Do NOT round "
            "any number, do NOT paraphrase any spec, and do NOT add any "
            "opinion, comparison, or recommendation — facts only."
        ),
        expected_output=(
            "A bulleted list, one bullet per product, each showing the "
            "exact product name, price_usd (as a plain integer, no "
            "rounding or currency-word paraphrasing), category, and specs "
            "string exactly as returned by the tool."
        ),
    )

    task_strategize = Task(
        description=(
            "Using ONLY the catalog facts produced above (do not invent "
            "any product or number not listed there), identify 2-3 "
            "distinct customer segments (e.g. budget-conscious students, "
            "portability-focused professionals, performance-focused power "
            "users). For each segment, recommend the best-fit laptop with "
            "one sentence of justification. Use the price_value_calculator "
            "tool to compute price_usd divided by the RAM in GB for at "
            "least two products, and cite that number to support your "
            "reasoning."
        ),
        expected_output=(
            "One short section per customer segment: segment name, "
            "recommended laptop, a one-sentence justification, and at "
            "least one computed price-per-GB-RAM value backing the choice."
        ),
        context=[task_analyze],
    )

    task_write = Task(
        description=(
            "Using ONLY the segment recommendations produced above (do "
            "not introduce any new fact, product, or number), write a "
            "concise, polished executive summary under 200 words for a "
            "back-to-school marketing campaign. Open with a one-sentence "
            "overview, one short paragraph per segment/recommended "
            "laptop, and close with a single named 'hero product' pick "
            "for the campaign."
        ),
        expected_output=(
            "A polished executive summary under 200 words, stakeholder-"
            "ready, ending with one clearly labeled hero product pick."
        ),
        context=[task_analyze, task_strategize],
    )

    return [task_analyze, task_strategize, task_write]
