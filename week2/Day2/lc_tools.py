"""
Week 2 / Day 2 — LangChain tool definitions.

Three tools, registered with LangChain's `@tool` decorator:

1. calculator          — reused concept from Day 1, reimplemented the
                          LangChain way.
2. get_weather         — reused concept from Day 1 (small fixed lookup
                          table, stands in for a real weather API).
3. get_product_price   — NEW: reads from a real external data source, a
                          local JSON "database" (products.json), so the
                          agent can answer questions that require looking
                          something up outside its own knowledge.

Docstrings matter here in a very literal way: LangChain feeds each tool's
docstring (plus its type-hinted signature) directly into the prompt the
model sees, in the same slot Day 1's hand-written `description` field
occupied. A vague or missing docstring is exactly as damaging here as a
vague `description` was in the raw Anthropic tool schema from Day 1 — the
model has no other source of truth about what the tool does, when to use
it, or what its inputs/outputs look like.
"""

import json
import os

from langchain_core.tools import tool

_PRODUCTS_PATH = os.path.join(os.path.dirname(__file__), "products.json")

with open(_PRODUCTS_PATH, "r", encoding="utf-8") as f:
    _PRODUCTS_DB = json.load(f)

_FAKE_WEATHER_DB = {
    "lahore": {"temp_c": 34, "condition": "hazy sunshine"},
    "karachi": {"temp_c": 31, "condition": "humid, partly cloudy"},
    "islamabad": {"temp_c": 27, "condition": "light rain"},
    "london": {"temp_c": 18, "condition": "overcast"},
    "dubai": {"temp_c": 41, "condition": "clear and very hot"},
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the numeric result.

    Use this any time a calculation, comparison of numbers, or price
    difference needs to be computed precisely instead of estimated.
    Only supports +, -, *, /, %, ** and parentheses.
    Input must be a valid arithmetic expression, e.g. '1499 - 549'.
    Do NOT pass words — only digits, operators, and parentheses.
    """
    allowed_chars = set("0123456789.+-*/()% \t")
    if not set(expression) <= allowed_chars:
        raise ValueError(
            f"Expression '{expression}' contains non-arithmetic characters. "
            "Refusing to evaluate for safety."
        )
    result = eval(expression, {"__builtins__": {}}, {})
    return str(result)


@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a named city.

    Returns the temperature in Celsius and a short condition string
    (e.g. 'sunny', 'rainy'). This is a stub backed by a small fixed
    lookup table (Lahore, Karachi, Islamabad, London, Dubai), not a
    live weather feed. If the city is not in the table, this raises an
    error rather than guessing — surface that error to the user instead
    of inventing a number.
    """
    key = city.strip().lower()
    if key not in _FAKE_WEATHER_DB:
        raise KeyError(
            f"No weather data available for '{city}'. Known cities: "
            f"{', '.join(c.title() for c in _FAKE_WEATHER_DB)}"
        )
    data = _FAKE_WEATHER_DB[key]
    return json.dumps({"city": city, **data})


@tool
def get_product_price(product_name: str) -> str:
    """Look up the price and specs of a laptop from the product catalog.

    This reads from a real external data source (products.json, a small
    local JSON 'database' of laptops), NOT from the model's own training
    knowledge — always call this tool instead of guessing a price.
    Known products: 'UltraBook Pro', 'BudgetBook Lite', 'Workstation Max',
    'AirCase 13'. Product name matching is case-insensitive.
    Raises an error if the product is not found in the catalog — surface
    that to the user rather than making up a price.
    """
    key = product_name.strip().lower()
    if key not in _PRODUCTS_DB:
        raise KeyError(
            f"No product found matching '{product_name}'. Known products: "
            f"{', '.join(p.title() for p in _PRODUCTS_DB)}"
        )
    data = _PRODUCTS_DB[key]
    return json.dumps({"product": product_name, **data})


TOOLS = [calculator, get_weather, get_product_price]
