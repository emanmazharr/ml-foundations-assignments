"""
Week 2 / Day 3 — plain data lookup used by the graph's "search" node.

Deliberately NOT wrapped in LangChain's @tool decorator this time: in a
LangGraph workflow, a node is just a Python function that reads and
returns state, so there's no agent deciding whether to call this — the
graph's structure itself decides when "search" runs. Keeping this as a
plain function (reused conceptually from Day 2's get_product_price)
avoids pulling in agent-executor machinery that isn't needed here.
"""

import json
import os

_PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "products.json")

with open(_PRODUCTS_PATH, "r", encoding="utf-8") as f:
    _PRODUCTS_DB = json.load(f)


def lookup_product(product_name: str) -> dict:
    """Look up a product's price/specs from the local JSON catalog.

    Raises KeyError if not found — callers should handle this explicitly
    rather than silently returning a placeholder.
    """
    key = product_name.strip().lower()
    if key not in _PRODUCTS_DB:
        raise KeyError(
            f"No product found matching '{product_name}'. Known products: "
            f"{', '.join(p.title() for p in _PRODUCTS_DB)}"
        )
    return {"product": product_name, **_PRODUCTS_DB[key]}
