# Week 3 Day 4 — LangGraph Integration (AFL Chat + Retrieval + Prediction)

## Run it now (no API key needed)
    python3 test_e2e_conversations.py     # 12 full conversations, all paths
    python3 test_router_accuracy.py       # 24-query router accuracy table
    python3 print_annotated_traces.py     # 3 annotated full state traces

## Run the real LangGraph + LLM version
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=...
    python3 afl_langgraph_production.py

## Data files expected (place alongside these scripts, or update the paths
at the top of afl_data_tools.py / afl_predict_tools.py):
    afl_players_info_raw.csv
    afl_players_round_by_round_stats_raw_-_afl_players_round_by_round_stats_raw_csv.csv
    team_matches_home_away_raw_-_team_matches_home_away_raw_csv.csv

## Files
    afl_data_tools.py             Task 2/3 (Day 3) - structured retrieval tools
    afl_guardrails.py             Task 1 (Day 3) - scope classifier + refusals
    afl_predict_tools.py          Task 3 (Day 4) - prediction models + tools
    afl_langgraph_offline.py      Tasks 1,2,4 (Day 4) - the graph (offline, executes here)
    afl_langgraph_production.py   Tasks 1,2,4 (Day 4) - real langgraph.StateGraph wiring
    test_router_accuracy.py       Task 2 - routing accuracy table
    test_e2e_conversations.py     Task 5 - 12 full conversations, all paths
    print_annotated_traces.py     Task 5 - 3 annotated full state traces
    day4_langgraph_report.md      Full written report for all 5 tasks
    AFL_Week3Day4_LangGraph_Notebook.ipynb   Everything above, executed, in one notebook
