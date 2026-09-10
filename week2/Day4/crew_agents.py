"""
Week 2 / Day 4 — the three specialized agents.

Business task decomposition (Task 1): "Review our laptop product catalog,
generate customer-segment insights, and write a stakeholder-ready summary
for a back-to-school marketing campaign" — split into three
non-overlapping roles:

1. Data Analyst    — ONLY touches raw catalog facts. No opinions, no
                      recommendations, no marketing language.
2. Market Strategist — ONLY interprets the analyst's facts into customer
                      segments and value-based recommendations. Never
                      re-fetches raw data itself.
3. Report Writer   — ONLY synthesizes the strategist's segment output
                      into a polished, concise executive summary. Never
                      introduces new facts or recommendations of its own.

Each responsibility boundary is enforced by BOTH the backstory/goal text
AND by which tools each agent does/doesn't have access to (see
crew_tools.py) — the two reinforce each other rather than relying on
prompt wording alone.
"""

import os

from crewai import Agent, LLM

from crew_tools import ListProductsTool, PriceValueCalculatorTool


def get_llm(model_name="gemini/gemini-3.5-flash-lite", temperature=0):
    """
    Uses gemini-3.5-flash-lite specifically: CrewAI's multi-agent, multi-task
    runs (especially hierarchical) make far more LLM calls per run than a
    single agent does, and the "flash-lite" tier's free-tier quota is
    substantially higher than the full "flash" model's — necessary headroom
    for running both a sequential AND a hierarchical crew in the same day.
    (Note: gemini-2.0-flash-lite, used in an earlier version of this file,
    was retired by Google on 2026-07-21 in favor of the 3.5 line — check
    https://ai.google.dev/gemini-api/docs/rate-limits for current quotas,
    since free-tier limits change over time.)
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Get a free key "
            "at https://aistudio.google.com/app/apikey."
        )
    return LLM(model=model_name, api_key=api_key, temperature=temperature)


def build_agents(llm):
    analyst = Agent(
        role="Product Data Analyst",
        goal=(
            "Extract precise, accurate facts and figures from the laptop "
            "product catalog to ground all downstream analysis. Never "
            "round numbers, never add opinions, never recommend anything."
        ),
        backstory=(
            "A meticulous data analyst who has seen too many marketing "
            "decisions go wrong because someone 'remembered' a price "
            "instead of checking it. You always cite the exact catalog "
            "figures and nothing else."
        ),
        tools=[ListProductsTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    strategist = Agent(
        role="Market Strategist",
        goal=(
            "Translate the analyst's raw catalog facts into 2-3 distinct "
            "customer-segment recommendations, each backed by a real "
            "computed value metric rather than a gut-feel guess."
        ),
        backstory=(
            "A strategist who has spent years positioning consumer "
            "electronics for different buyer personas — students, "
            "professionals, power users — and always grounds a "
            "recommendation in a specific number, not just intuition."
        ),
        tools=[PriceValueCalculatorTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Stakeholder Report Writer",
        goal=(
            "Produce a concise, polished, stakeholder-ready executive "
            "summary (under 200 words) recommending which laptops to "
            "feature in the campaign, based ONLY on the strategist's "
            "segment recommendations."
        ),
        backstory=(
            "A communications specialist known for distilling analytical "
            "and strategic input into a one-page summary a busy "
            "stakeholder can act on in under a minute — never inventing "
            "new facts, only reorganizing and polishing what's already "
            "been decided upstream."
        ),
        tools=[],  # deliberately no tools — pure synthesis role
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    return analyst, strategist, writer


def build_manager(llm):
    """A distinct manager persona for the hierarchical crew (Task 4) —
    used as `manager_agent` rather than just `manager_llm`, so the crew
    has an actual manager persona that delegates and reviews, not just an
    anonymous default coordinator."""
    return Agent(
        role="Campaign Project Manager",
        goal=(
            "Coordinate the data analyst, market strategist, and report "
            "writer to produce an accurate, compelling stakeholder-ready "
            "marketing summary — delegating each piece of work to the "
            "right specialist and reviewing their output before it moves "
            "forward, rather than doing the specialist work yourself."
        ),
        backstory=(
            "An experienced project manager who has run cross-functional "
            "marketing campaigns for years. You know exactly which "
            "specialist owns which piece of work, and you never let a "
            "low-quality draft move forward without sending it back for "
            "revision."
        ),
        llm=llm,
        allow_delegation=True,
        verbose=True,
    )
