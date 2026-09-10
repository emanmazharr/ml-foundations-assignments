"""
Week 2 / Day 4 — Crew assembly: sequential (Task 3) and hierarchical
(Task 4) versions of the same underlying work.
"""

from crewai import Crew, Process

from crew_agents import build_agents, build_manager
from crew_tasks import build_sequential_tasks, build_hierarchical_tasks


def build_sequential_crew(llm):
    analyst, strategist, writer = build_agents(llm)
    tasks = build_sequential_tasks(analyst, strategist, writer)
    return Crew(
        agents=[analyst, strategist, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )


def build_hierarchical_crew(llm):
    analyst, strategist, writer = build_agents(llm)
    manager = build_manager(llm)
    tasks = build_hierarchical_tasks()
    return Crew(
        agents=[analyst, strategist, writer],
        tasks=tasks,
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=True,
    )
