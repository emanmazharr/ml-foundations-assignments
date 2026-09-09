"""
Week 2 / Day 3 — State schema for a self-correcting product-recommendation
research assistant.

Scenario: given a question comparing two laptops, the graph searches a
product catalog, drafts a recommendation, critiques its own draft, loops
back to redraft if the critique score is too low, then pauses for human
approval before "sending" the recommendation to a client (the risky
action) and finally formats the result.
"""

import operator
from typing import Annotated, Optional, TypedDict

from pydantic import BaseModel, Field


class State(TypedDict):
    question: str                     # the user's original question
    product_a: str                    # first product name to compare
    product_b: str                    # second product name to compare
    search_data: str                  # raw catalog lookup results (JSON text)
    draft: str                        # current draft recommendation
    critique_feedback: str            # most recent critique's feedback text
    quality_score: int                # most recent critique's 1-10 score
    retries: int                      # how many redraft cycles have run
    max_retries: int                  # hard cap to prevent infinite loops
    approved: Optional[bool]          # human-in-the-loop decision, None until set
    final_output: str                 # formatted final answer
    # `log` accumulates across nodes/cycles rather than being overwritten,
    # via the `operator.add` reducer — this is what lets us print a full
    # trace of every pass through the self-correction loop afterward.
    log: Annotated[list[str], operator.add]


class CritiqueResult(BaseModel):
    """Structured output for the critique node (Task 5's structured-output
    idea, reused here to avoid brittle string-parsing of a pass/fail
    decision)."""

    score: int = Field(description="Quality score from 1 (poor) to 10 (excellent)")
    feedback: str = Field(description="1-2 sentences of specific, actionable feedback")
    should_revise: bool = Field(description="True if the draft needs another pass")
