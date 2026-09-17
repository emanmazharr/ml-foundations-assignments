"""
afl_langgraph_offline.py
--------------------------
Day 4: Task 1-4 - an executable stand-in for the LangGraph application.

HONESTY NOTE: this sandbox has no network access and `langgraph` is not
installed here (verified - no pip access). This module hand-implements
the SAME graph shape LangGraph would run (explicit State dict, node
functions, a router that returns the next node name, conditional
branching) so Tasks 1-5 have real, executed evidence. The literal
`langgraph.graph.StateGraph` wiring - same nodes, same edges - is in
afl_langgraph_production.py for the user's own environment
(`pip install langgraph langchain-anthropic`, set an API key).

STATE SCHEMA (Task 1)
----------------------
    user_query              : str                       - this turn's raw text
    conversation_history    : list[(role, text)]         - full transcript so far
    memory                  : dict                       - slot memory (last team/
                                                            player/season) carried
                                                            across turns for follow-ups
    intent                   : "off_topic" | "retrieval" | "prediction" | "clarification"
    tool_name                : str | None                 - which tool the node called
    tool_args                : dict | None
    tool_result              : dict | None                - raw return value of the tool
    tool_error                : str | None                - set on LookupError_ / PredictionInputError
    validation_passed         : bool | None
    final_response            : str

GRAPH SHAPE (Task 1)
----------------------
    START
      -> router_node                (classifies intent; off-topic -> refusal_node)
           -> retrieval_node         (factual/stat questions -> afl_data_tools)
           -> prediction_node        (win/top-scorer questions -> afl_predict_tools)
           -> refusal_node           (off-topic -> guardrail message, skips validation)
      -> validation_node             (retrieval/prediction only: did the tool
                                       succeed? did it need a team/player we
                                       couldn't resolve?)
           -> response_formatter_node   (success -> phrase the tool_result)
           -> clarification_node        (resolution failure -> ask, don't guess)
    -> END (final_response set)

WHY EXPLICIT ROUTING, NOT ONE FREE AGENT (Task 1 justification)
-------------------------------------------------------------------
A single generic tool-calling agent decides, per turn, whether to call a
retrieval tool, a prediction tool, both, or neither - and that decision
lives inside a black-box LLM call that can drift (e.g. quietly answering
a win-probability question from "general football knowledge" instead of
calling predict_match_winner, or forgetting the required disclaimer on a
prediction because nothing structurally forces it). Routing explicitly:
  1. Makes the prediction/retrieval boundary an enforced program branch,
     not a hope - a prediction can only reach the user through
     prediction_node, which always attaches the probabilistic disclaimer
     and the top-feature explanation, every single time.
  2. Puts a real validation_node between "tool ran" and "user sees an
     answer" - a monolithic agent can just narrate over a tool error;
     here a tool_error hard-routes to clarification_node instead.
  3. Makes failure modes inspectable and independently testable (this
     is exactly what Task 2's routing-accuracy table and Task 5's
     annotated traces below are checking) - you can unit-test the
     router node in isolation, which you cannot do to "the agent's
     judgement" inside one big prompt.
"""

import re
from typing import TypedDict, List, Tuple, Optional, Any, Dict

from afl_guardrails import classify_scope, refusal_response
import afl_data_tools as dt
import afl_predict_tools as pred


class AFLGraphState(TypedDict, total=False):
    user_query: str
    conversation_history: List[Tuple[str, str]]
    memory: Dict[str, Any]
    intent: str
    tool_name: Optional[str]
    tool_args: Optional[dict]
    tool_result: Optional[dict]
    tool_error: Optional[str]
    validation_passed: Optional[bool]
    final_response: str
    trace: List[dict]   # step-by-step log for Task 5 annotated traces


PREDICTION_PATTERNS = [
    r"\bwho.?ll win\b", r"\bwho will win\b", r"\bwill .* beat\b", r"\bwill .* win\b",
    r"\bpredict\w*\b", r"\bwho.?s going to win\b", r"\bgoing to win\b", r"\bchances of winning\b",
    r"\bwin probability\b", r"\btop.{0,20}scor(e|er|ers|ing)\b.*(next|this|upcoming|round)",
    r"\btop predicted\b", r"\bpredicted (top )?scorer", r"\bwho will top.?score\b",
    r"\bforecast\b", r"\bodds\b", r"\bfinals chances\b", r"\bchances\b",
]


def router_node(state: AFLGraphState) -> AFLGraphState:
    text = state["user_query"]
    context = [x for role, x in state["conversation_history"] if role == "user"][-4:]
    scope = classify_scope(text, context)
    step = {"node": "router_node", "input": text, "scope_classification": scope}

    if not scope["in_scope"]:
        state["intent"] = "off_topic"
        step["decision"] = "off_topic"
        state["trace"].append(step)
        return state

    t = text.lower()
    if any(re.search(p, t) for p in PREDICTION_PATTERNS):
        state["intent"] = "prediction"
        step["decision"] = "prediction"
    else:
        state["intent"] = "retrieval"
        step["decision"] = "retrieval"
    state["trace"].append(step)
    return state


def refusal_node(state: AFLGraphState) -> AFLGraphState:
    scope = state["trace"][-1]["scope_classification"]
    reply = refusal_response(scope)
    state["final_response"] = reply
    state["validation_passed"] = None
    state["trace"].append({"node": "refusal_node", "output": reply})
    return state


# --------------------------------------------------------------- helpers --
def _player_from_text(text, memory):
    m = re.search(r"(?:player[_ ]?(?:id)?|pid)\s*#?\s*(\d{3,6})", text, re.I)
    if m:
        try:
            return dt.resolve_player(int(m.group(1)))
        except dt.LookupError_:
            pass
    text_l = text.lower()
    best = None
    for name_l, pid in dt.NAME_TO_ID.items():
        if name_l in text_l and (best is None or len(name_l) > len(best[0])):
            best = (name_l, pid)
    if best:
        return dt.resolve_player(int(best[1]))
    if memory.get("last_player_id"):
        return {"player_id": memory["last_player_id"], "player_name": memory.get("last_player_name")}
    return None


def _teams_from_text(text):
    text_l = text.lower()
    # 1) exact/nickname hits first
    hits = [t for t in dt.ALL_TEAMS if t.lower() in text_l]
    nick_hits = [pred.NICKNAME_MAP[k] for k in pred.NICKNAME_MAP if re.search(rf"\b{re.escape(k)}\b", text_l)]
    combined = list(dict.fromkeys(hits + nick_hits))
    if len(combined) >= 2:
        return combined[:2]
    # 2) fall back to partial word match, e.g. "Carlton" -> "Carlton Blues"
    for t in dt.ALL_TEAMS:
        for word in t.split():
            if len(word) > 3 and re.search(rf"\b{re.escape(word.lower())}\b", text_l) and t not in combined:
                combined.append(t)
                break
        if len(combined) >= 2:
            break
    return combined[:2]


def _season_from_text(text, memory):
    m = re.search(r"\b(19|20)\d{2}\b", text)
    return int(m.group(0)) if m else memory.get("last_season")


# ------------------------------------------------------------ retrieval --
def retrieval_node(state: AFLGraphState) -> AFLGraphState:
    text = state["user_query"]
    t = text.lower()
    memory = state["memory"]
    step = {"node": "retrieval_node", "input": text}
    try:
        if re.search(r"head.to.head|record (vs|against)|played each other", t):
            teams = _teams_from_text(text)
            if len(teams) < 2:
                raise dt.LookupError_("need_two_teams")
            res = dt.get_team_record_vs_opponent(teams[0], teams[1])
            state["tool_name"], state["tool_args"] = "get_team_record_vs_opponent", {"team": teams[0], "opponent": teams[1]}
            memory.update(last_team=teams[0], last_opponent=teams[1])

        elif re.search(r"career average|compare.*career", t):
            who = _player_from_text(text, memory)
            if not who:
                raise dt.LookupError_("need_player")
            stat = "disposals" if "disposal" in t else ("goals" if "goal" in t else "fantasy_points")
            res = dt.compare_player_recent_to_career(who["player_id"], stat=stat, recent_n=5)
            state["tool_name"], state["tool_args"] = "compare_player_recent_to_career", {"player": who["player_id"], "stat": stat}
            memory.update(last_player_id=who["player_id"], last_player_name=who["player_name"])

        elif re.search(r"round before|game before|earlier round|previous round", t):
            who = _player_from_text(text, memory)
            if not who:
                raise dt.LookupError_("need_player")
            old = memory.get("last_games", [{}])[0] if memory.get("last_games") else None
            if old and "year" in old:
                res = dt.get_player_recent_rounds(who["player_id"], 1, before_year=old["year"], before_round=old["round"])
            else:
                res = dt.get_player_recent_rounds(who["player_id"], 1)
            state["tool_name"], state["tool_args"] = "get_player_recent_rounds", {"player": who["player_id"]}
            memory.update(last_player_id=who["player_id"], last_player_name=who["player_name"], last_games=res["games"])

        elif re.search(r"top player|leaderboard|leading player", t):
            teams = _teams_from_text(text)
            season = _season_from_text(text, memory)
            if not teams or not season:
                raise dt.LookupError_("need_team_and_season")
            res = dt.get_team_top_players(teams[0], season)
            state["tool_name"], state["tool_args"] = "get_team_top_players", {"team": teams[0], "season": season}
            memory.update(last_team=teams[0], last_season=season)

        elif re.search(r"season stats|in (19|20)\d{2}", t):
            who = _player_from_text(text, memory)
            season = _season_from_text(text, memory)
            if not who or not season:
                raise dt.LookupError_("need_player_and_season")
            res = dt.get_player_season_stats(who["player_id"], season)
            state["tool_name"], state["tool_args"] = "get_player_season_stats", {"player": who["player_id"], "season": season}
            memory.update(last_player_id=who["player_id"], last_player_name=who["player_name"], last_season=season)

        elif re.search(r"current form|recent form|how (are|is) .* going|last \d+ (games|matches)", t) or \
             (_teams_from_text(text) and not _player_from_text(text, memory)):
            teams = _teams_from_text(text)
            if not teams:
                raise dt.LookupError_("need_team")
            res = dt.get_team_recent_form(teams[0], n=5)
            state["tool_name"], state["tool_args"] = "get_team_recent_form", {"team": teams[0]}
            memory["last_team"] = teams[0]

        else:
            who = _player_from_text(text, memory)
            if not who:
                raise dt.LookupError_("need_player_or_team")
            n = 1 if "last round" in t else 5
            res = dt.get_player_recent_rounds(who["player_id"], n=n)
            state["tool_name"], state["tool_args"] = "get_player_recent_rounds", {"player": who["player_id"], "n": n}
            memory.update(last_player_id=who["player_id"], last_player_name=who["player_name"], last_games=res["games"])

        state["tool_result"] = res
        state["tool_error"] = None
        step["tool_name"] = state["tool_name"]
        step["tool_result"] = res
    except dt.LookupError_ as e:
        state["tool_result"] = None
        state["tool_error"] = str(e)
        step["tool_error"] = str(e)

    state["trace"].append(step)
    return state


# ----------------------------------------------------------- prediction --
def prediction_node(state: AFLGraphState) -> AFLGraphState:
    text = state["user_query"]
    t = text.lower()
    step = {"node": "prediction_node", "input": text}
    try:
        if re.search(r"top.{0,20}scor|top player|top predicted", t):
            teams = _teams_from_text(text)
            if not teams:
                raise pred.PredictionInputError("need_team")
            resolved_date = pred.resolve_relative_date(text)
            stat_type = "fantasy_points"
            for kw in ("tackle", "mark", "clearance", "goal", "disposal"):
                if kw in t:
                    stat_type = kw + "s"
                    break
            res = pred.predict_top_player(teams[0], stat_type=stat_type)
            state["tool_name"], state["tool_args"] = "predict_top_player", {"team": teams[0], "as_of": resolved_date, "stat_type": stat_type}
        else:
            teams = _teams_from_text(text)
            memory = state["memory"]
            if len(teams) < 2:
                mem_teams = [memory.get("last_team"), memory.get("last_opponent")]
                mem_teams = [x for x in mem_teams if x]
                for mt in mem_teams:
                    if mt not in teams:
                        teams.append(mt)
            if len(teams) < 2:
                raise pred.PredictionInputError("need_two_teams")
            resolved_date = pred.resolve_relative_date(text)
            res = pred.predict_match_winner(teams[0], teams[1], resolved_date)
            state["tool_name"], state["tool_args"] = "predict_match_winner", {
                "team_a": teams[0], "team_b": teams[1], "date": resolved_date
            }
        state["tool_result"] = res
        state["tool_error"] = None
        step["tool_name"] = state["tool_name"]
        step["tool_result"] = res
    except pred.PredictionInputError as e:
        state["tool_result"] = None
        state["tool_error"] = str(e)
        step["tool_error"] = str(e)

    state["trace"].append(step)
    return state


# ----------------------------------------------------------- validation --
def validation_node(state: AFLGraphState) -> AFLGraphState:
    ok = state.get("tool_error") is None and state.get("tool_result") is not None
    state["validation_passed"] = ok
    state["trace"].append({"node": "validation_node", "passed": ok, "error": state.get("tool_error")})
    return state


def clarification_node(state: AFLGraphState) -> AFLGraphState:
    err = state.get("tool_error", "")
    prompts = {
        "need_two_teams": "Which two teams did you mean? Nicknames like 'Pies' or 'Cats' are fine.",
        "need_player": "Which player did you mean?",
        "need_team": "Which team did you mean?",
        "need_team_and_season": "Which team and which season?",
        "need_player_and_season": "Which player, and which season?",
        "need_player_or_team": "Could you name a specific player or team?",
    }
    if err in prompts:
        reply = prompts[err]
    elif "not modelled" in err or "not supported" in err:
        # Task 4 fallback path: unsupported request, say so plainly - don't guess
        reply = f"I can't do that one yet: {err} I won't guess a number for something I don't model."
    else:
        reply = f"I couldn't resolve that request ({err}) - could you clarify the team, player, or season?"
    state["final_response"] = reply
    state["trace"].append({"node": "clarification_node", "output": reply})
    return state


# --------------------------------------------------- response formatting --
def response_formatter_node(state: AFLGraphState) -> AFLGraphState:
    res = state["tool_result"]
    name = state["tool_name"]

    if name == "get_team_record_vs_opponent":
        reply = (f"Across {res['games_played']} recorded meetings, {res['team']} have "
                 f"{res['wins_for_team']} wins and {res['losses_for_team']} losses "
                 f"(plus {res['draws']} draw(s)) against {res['opponent']}. Most recent: "
                 f"{res['most_recent_score']} ({res['most_recent_result']} for {res['team']}).")
    elif name == "compare_player_recent_to_career":
        reply = (f"{res['player_name']}'s last-5 {res['stat'].replace('_',' ')} average is "
                 f"{res['recent_avg']}, vs a career average of {res['career_avg']} over "
                 f"{res['career_games']} games.")
    elif name == "get_player_recent_rounds":
        g = res["games"][-1]
        reply = (f"{res['player_name']} - {g['year']} round {g['round']} vs {g['opponent']} "
                 f"({g['result']}): {g['disposals']} disposals, {g['goals']} goals, "
                 f"{g['fantasy_points']} fantasy points.")
    elif name == "get_team_top_players":
        lines = ", ".join(f"{l['player_name']} ({l['avg_fantasy_points']})" for l in res["leaders"])
        reply = f"Top fantasy-point scorers for {res['team']} in {res['season']}: {lines}."
    elif name == "get_player_season_stats":
        reply = (f"In {res['season']}, {res['player_name']} played {res['games_played']} games: "
                 f"{res['avg_disposals']} disposals/game, {res['avg_goals']} goals/game, "
                 f"{res['avg_fantasy_points']} fantasy points/game.")
    elif name == "get_team_recent_form":
        last = res["games"][-1]
        reply = (f"Over {res['team']}'s last {res['n_returned']} matches they've won "
                 f"{res['win_rate_pct']}% of the time; most recently vs {last['opponent']} "
                 f"({last['result']}, {last['team_score']}-{last['opponent_score']}).")
    elif name == "predict_match_winner":
        winner_team = res["home_team"] if res["predicted_winner"] == "W" else (
            res["away_team"] if res["predicted_winner"] == "L" else "a draw")
        reply = (f"Model estimate: {winner_team} most likely (home win {res['probabilities'].get('W',0)*100:.0f}%, "
                 f"away win {res['probabilities'].get('L',0)*100:.0f}%, draw "
                 f"{res['probabilities'].get('D',0)*100:.0f}%). Driven mainly by: "
                 f"{'; '.join(res['top_features'])}. {res['disclaimer']}")
    elif name == "predict_top_player":
        lines = ", ".join(f"{l['player_name']} ({l['predicted_fantasy_points']})" for l in res["leaders"])
        reply = (f"Predicted top fantasy scorers for {res['team']}'s next match: {lines}. "
                 f"Driven mainly by: {', '.join(res['top_features'])}. {res['disclaimer']}")
    else:
        reply = "Here's what I found: " + str(res)

    state["final_response"] = reply
    state["trace"].append({"node": "response_formatter_node", "output": reply})
    return state


# -------------------------------------------------------------- the graph --
def run_graph(user_query: str, conversation_history: list, memory: dict) -> AFLGraphState:
    """Executes the node sequence described in the module docstring,
    exactly the shape a real langgraph.graph.StateGraph would run."""
    state: AFLGraphState = {
        "user_query": user_query,
        "conversation_history": conversation_history,
        "memory": memory,
        "trace": [],
    }
    state = router_node(state)

    if state["intent"] == "off_topic":
        state = refusal_node(state)
        return state

    if state["intent"] == "prediction":
        state = prediction_node(state)
    else:
        state = retrieval_node(state)

    state = validation_node(state)
    if state["validation_passed"]:
        state = response_formatter_node(state)
    else:
        state = clarification_node(state)
    return state


class AFLGraphAgent:
    """Thin session wrapper so multi-turn conversations carry memory +
    history the same way the Day-3 offline agent did."""
    def __init__(self):
        self.memory = {}
        self.history = []

    def chat(self, text: str) -> str:
        final_state = run_graph(text, self.history, self.memory)
        self.history.append(("user", text))
        self.history.append(("assistant", final_state["final_response"]))
        self.last_state = final_state
        return final_state["final_response"]
