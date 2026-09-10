"""
Week 2 / Day 4 — CrewAI tools.

Two tools, deliberately given to only the ONE agent each that actually
needs them (Task 2's "keep tool access role-appropriate" requirement):

- `ListProductsTool`  -> given ONLY to the Data Analyst. It's the sole
                          source of ground-truth catalog data; no other
                          agent should be able to invent or re-fetch it.
- `PriceValueCalculatorTool` -> given ONLY to the Market Strategist, who
                          needs to back a segment recommendation with a
                          real computed number (price per GB of RAM)
                          rather than an eyeballed guess.

The Report Writer gets NO tools at all — its job is purely to synthesize
text already produced by the other two agents, and giving it tool access
would just create an opportunity to re-fetch/re-compute things and
introduce inconsistency with what was already reviewed upstream.
"""

import json
import os
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "products.json")

with open(_PRODUCTS_PATH, "r", encoding="utf-8") as f:
    _PRODUCTS_DB = json.load(f)


class ListProductsTool(BaseTool):
    name: str = "list_products"
    description: str = (
        "Return the FULL laptop product catalog as JSON (every product's "
        "exact price_usd, category, and specs). This is the only source "
        "of ground-truth product data — always call this instead of "
        "recalling or estimating any price or spec from memory."
    )

    def _run(self) -> str:
        return json.dumps(_PRODUCTS_DB, indent=2)


class CalculatorInput(BaseModel):
    expression: str = Field(
        description="A valid arithmetic expression, e.g. '1499 / 16' for price per GB of RAM."
    )


class PriceValueCalculatorTool(BaseTool):
    name: str = "price_value_calculator"
    description: str = (
        "Evaluate a basic arithmetic expression, e.g. to compute a "
        "price-per-GB-of-RAM value metric for a laptop (price_usd / "
        "ram_gb). Only supports +, -, *, /, %, ** and parentheses. Use "
        "this instead of estimating a ratio in your head."
    )
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        allowed_chars = set("0123456789.+-*/()% \t")
        if not set(expression) <= allowed_chars:
            raise ValueError(
                f"Expression '{expression}' contains non-arithmetic characters."
            )
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
