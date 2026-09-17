"""
afl_data_tools.py
------------------
Task 2: structured, dataset-grounded lookup functions for the AFL chat agent.

DATA SOURCES (all provided CSVs, no scraping/invention):
  - afl_players_info_raw.csv                    -> player_id -> name mapping
  - afl_players_round_by_round_stats_raw_*.csv  -> one row per player per
                                                    match: disposals, goals,
                                                    fantasy_points, result,
                                                    plus 25+ other raw stats
  - team_matches_home_away_raw_*.csv            -> one row per team per
                                                    match: score, opponent,
                                                    result (W/L/D), margin,
                                                    venue, crowd

DESIGN PRINCIPLE (structured vs semantic retrieval):
Every one of these files is fully structured/tabular with exact numeric
columns. There is no free-text match-report or commentary data supplied,
so there is nothing to put in a vector store - a semantic/embedding
search over these tables would be strictly worse than a pandas filter:
it could return the "closest sounding" row instead of the exact one, and
sports fans / bettors care about the *exact* number, not an approximate
match. So Task 2's entire retrieval layer is structured lookups only,
each implemented as a plain pandas query with no LLM involved in
computing the number - only in phrasing the sentence around it.
"""

import re
import pandas as pd
from difflib import get_close_matches

import os
_DATA_DIR = os.getenv("AFL_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "afl_datasets"))
INFO_CSV = os.path.join(_DATA_DIR, "afl_players_info_raw.csv")
RBR_CSV = os.path.join(_DATA_DIR, "afl_players_round_by_round_stats_raw - afl_players_round_by_round_stats_raw.csv.csv")
TEAM_MATCHES_CSV = os.path.join(_DATA_DIR, "team_matches_home_away_raw - team_matches_home_away_raw.csv.csv")

# ------------------------------------------------------------------ load ---
info_df = pd.read_csv(INFO_CSV)
info_df = info_df.drop_duplicates(subset=["id"]).set_index("id")

rbr_df = pd.read_csv(RBR_CSV, low_memory=False, parse_dates=["match_date"])
rbr_df = rbr_df.drop_duplicates(subset=["id"])

team_matches_df = pd.read_csv(TEAM_MATCHES_CSV, parse_dates=["match_date"])
# clean up messy raw team names: strip whitespace, fix inconsistent casing,
# and normalise a known alias ("W. Bulldogs" -> "Western Bulldogs")
_ALIAS = {"w. bulldogs": "Western Bulldogs"}
_CANONICAL = {}
for raw in pd.concat([team_matches_df["team_name"], team_matches_df["opponent"]]).dropna().unique():
    key = raw.strip().lower()
    canonical = _ALIAS.get(key, raw.strip())
    # prefer a Title Case-looking canonical form (the raw data mixes
    # 'Carlton Blues' and 'carlton blues' for the same team)
    if key not in _CANONICAL or (canonical[:1].isupper() and not _CANONICAL[key][:1].isupper()):
        _CANONICAL[key] = canonical


def _clean_team_col(series: pd.Series) -> pd.Series:
    return series.str.strip().str.lower().map(_CANONICAL)


team_matches_df["team_name"] = _clean_team_col(team_matches_df["team_name"])
team_matches_df["opponent"] = _clean_team_col(team_matches_df["opponent"])

ALL_TEAMS = sorted(set(rbr_df["team"]) | set(rbr_df["opponent"]) | set(team_matches_df["team_name"]))
NAME_TO_ID = {}
for pid, row in info_df.iterrows():
    candidates = {row.get("player_name")}
    fullname = row.get("player_full_name")
    if isinstance(fullname, str):
        candidates.add(fullname.replace("_", " "))
    for nm in candidates:
        if isinstance(nm, str) and nm.strip():
            NAME_TO_ID.setdefault(nm.strip().lower(), pid)


class LookupError_(Exception):
    """Raised when a requested entity isn't in the dataset - the agent
    must surface this as 'I don't have that in my data', never guess."""
    pass


# --------------------------------------------------------------- helpers ---
def resolve_team(name: str) -> str:
    """Resolve a user-typed team name/nickname to the exact team string
    used in the dataset, via case-insensitive exact/substring match,
    then a fuzzy fallback. Raises LookupError_ if nothing close enough."""
    if not name:
        raise LookupError_("No team name given.")
    name_l = name.strip().lower()
    for t in ALL_TEAMS:
        if t.lower() == name_l:
            return t
    substr_hits = [t for t in ALL_TEAMS if name_l in t.lower()]
    if len(substr_hits) == 1:
        return substr_hits[0]
    if len(substr_hits) > 1:
        raise LookupError_(f"'{name}' matches multiple teams: {substr_hits}. Please be more specific.")
    close = get_close_matches(name, ALL_TEAMS, n=1, cutoff=0.6)
    if close:
        return close[0]
    raise LookupError_(f"No team matching '{name}' found in the dataset.")


def resolve_player(name_or_id) -> dict:
    """Resolve a player name (fuzzy, case-insensitive) OR a numeric
    player_id to {'player_id': int, 'player_name': str}. This is what
    lets the agent take natural questions ('how is Marcus Bontempelli
    going?') instead of forcing users to know internal numeric ids."""
    if isinstance(name_or_id, (int, float)) or (isinstance(name_or_id, str) and name_or_id.strip().isdigit()):
        pid = int(name_or_id)
        if pid in info_df.index:
            return {"player_id": pid, "player_name": info_df.loc[pid, "player_name"]}
        if pid in set(rbr_df["player_id"]):
            return {"player_id": pid, "player_name": f"Player {pid} (name not on file)"}
        raise LookupError_(f"No player with id {pid} in the dataset.")

    name_l = str(name_or_id).strip().lower()
    if name_l in NAME_TO_ID:
        pid = NAME_TO_ID[name_l]
        return {"player_id": int(pid), "player_name": info_df.loc[pid, "player_name"]}
    close = get_close_matches(name_l, list(NAME_TO_ID.keys()), n=1, cutoff=0.72)
    if close:
        pid = NAME_TO_ID[close[0]]
        return {"player_id": int(pid), "player_name": info_df.loc[pid, "player_name"]}
    raise LookupError_(f"No player matching '{name_or_id}' found in the dataset.")


# ---------------------------------------------------------------------
# TOOL 0: structured lookup - player profile/bio
# ---------------------------------------------------------------------
def get_player_profile(player) -> dict:
    """Exact lookup of a player's biographical info (debut date, height,
    weight, teams played for) straight from afl_players_info_raw."""
    who = resolve_player(player)
    pid = who["player_id"]
    if pid not in info_df.index:
        raise LookupError_(f"No profile info on file for {who['player_name']} (id {pid}).")
    row = info_df.loc[pid]
    return {
        "player_id": pid,
        "player_name": row["player_name"],
        "debut_date": row.get("debut_date"),
        "last_date": row.get("last_date"),
        "height_cm": row.get("height"),
        "weight_kg": row.get("weight"),
        "teams": row.get("player_teams"),
    }


# ---------------------------------------------------------------------
# TOOL 1: structured lookup - team head-to-head record (real W/L/D)
# ---------------------------------------------------------------------
def get_team_record_vs_opponent(team: str, opponent: str) -> dict:
    """Exact head-to-head record between two teams: wins/losses/draws and
    average winning margin, computed directly from every recorded match
    between them in team_matches_home_away_raw (result column is the
    actual final result of that game, not a model estimate)."""
    a = resolve_team(team)
    b = resolve_team(opponent)
    sub = team_matches_df[(team_matches_df["team_name"] == a) & (team_matches_df["opponent"] == b)]
    if sub.empty:
        raise LookupError_(f"No recorded matches between {a} and {b}.")
    wins = int((sub["result"] == "W").sum())
    losses = int((sub["result"] == "L").sum())
    draws = int((sub["result"] == "D").sum())
    last = sub.sort_values("match_date").iloc[-1]
    return {
        "team": a,
        "opponent": b,
        "games_played": int(len(sub)),
        "wins_for_team": wins,
        "losses_for_team": losses,
        "draws": draws,
        "avg_margin_for_team": round(float(sub["margin"].mean()), 1),
        "most_recent_match_date": str(last["match_date"].date()),
        "most_recent_result": last["result"],
        "most_recent_score": f"{a} {last['team_score']} - {last['opponent_score']} {b}",
    }


# ---------------------------------------------------------------------
# TOOL 2: structured lookup - team current/recent form
# ---------------------------------------------------------------------
def get_team_recent_form(team: str, n: int = 5) -> dict:
    """Exact lookup of a team's last n recorded matches (date, opponent,
    result, score, margin, venue), plus win rate over that window,
    straight from team_matches_home_away_raw."""
    a = resolve_team(team)
    sub = team_matches_df[team_matches_df["team_name"] == a].sort_values("match_date")
    if sub.empty:
        raise LookupError_(f"No matches on file for {a}.")
    recent = sub.tail(int(n))
    win_rate = round(float((recent["result"] == "W").mean()) * 100, 1)
    games = recent[["match_date", "opponent", "result", "team_score", "opponent_score", "margin", "venue"]].copy()
    games["match_date"] = games["match_date"].dt.date.astype(str)
    return {
        "team": a,
        "n_requested": int(n),
        "n_returned": int(len(games)),
        "win_rate_pct": win_rate,
        "games": games.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------
# TOOL 3: structured lookup - player season stats (real name resolution)
# ---------------------------------------------------------------------
def get_player_season_stats(player, season: int) -> dict:
    """Exact aggregation of a player's games/totals/averages for one
    season (disposals, goals, fantasy points), computed directly from
    their raw match rows in the round-by-round table. `player` may be a
    name (fuzzy-matched) or a numeric player_id."""
    who = resolve_player(player)
    pid = who["player_id"]
    sub = rbr_df[(rbr_df["player_id"] == pid) & (rbr_df["year"] == int(season))]
    if sub.empty:
        raise LookupError_(f"No rows for {who['player_name']} (id {pid}) in season {season}.")
    return {
        "player_id": pid,
        "player_name": who["player_name"],
        "season": int(season),
        "games_played": int(len(sub)),
        "team(s)": sorted(sub["team"].unique().tolist()),
        "total_disposals": float(sub["disposals"].sum(skipna=True)),
        "avg_disposals": round(float(sub["disposals"].mean(skipna=True)), 2),
        "total_goals": float(sub["goals"].sum(skipna=True)),
        "avg_goals": round(float(sub["goals"].mean(skipna=True)), 2),
        "avg_fantasy_points": round(float(sub["fantasy_points"].mean(skipna=True)), 2),
        "wins": int((sub["result"] == "W").sum()),
        "losses": int((sub["result"] == "L").sum()),
    }


# ---------------------------------------------------------------------
# TOOL 4: structured lookup - player's last N rounds (round-by-round log)
# ---------------------------------------------------------------------
def get_player_recent_rounds(player, n: int = 5, before_year: int = None, before_round=None) -> dict:
    """Exact lookup of a player's most recent N match rows (round-by-
    round disposals/goals/fantasy points/result), optionally as of a
    point in time (before_year/before_round) so follow-ups like 'the
    round before that' can walk further back through this same log."""
    who = resolve_player(player)
    pid = who["player_id"]
    sub = rbr_df[rbr_df["player_id"] == pid].copy()
    sub["_round_sort"] = pd.to_numeric(sub["round"], errors="coerce").fillna(99)
    sub = sub.sort_values(["year", "match_date"])
    if before_year is not None:
        if before_round is not None:
            br = pd.to_numeric(pd.Series([before_round]), errors="coerce").fillna(99).iloc[0]
            sub = sub[(sub["year"] < before_year) | ((sub["year"] == before_year) & (sub["_round_sort"] < br))]
        else:
            sub = sub[sub["year"] < before_year]
    if sub.empty:
        raise LookupError_(f"No match rows found for {who['player_name']} matching that time window.")
    sub = sub.tail(int(n))
    cols = ["year", "round", "team", "opponent", "match_date", "disposals", "goals", "fantasy_points", "result"]
    games = sub[cols].copy()
    games["match_date"] = games["match_date"].dt.date.astype(str)
    return {
        "player_id": pid,
        "player_name": who["player_name"],
        "n_requested": int(n),
        "n_returned": int(len(games)),
        "games": games.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------
# TOOL 5: structured lookup - compare recent form to career average
# ---------------------------------------------------------------------
def compare_player_recent_to_career(player, stat: str = "disposals", recent_n: int = 5) -> dict:
    """Exact comparison of a player's average for `stat` over their most
    recent `recent_n` games vs their full career average for that stat,
    both computed directly from the raw rows (no model, no estimate)."""
    if stat not in ("disposals", "goals", "fantasy_points"):
        raise LookupError_(f"Unsupported stat '{stat}'. Choose from disposals, goals, fantasy_points.")
    who = resolve_player(player)
    pid = who["player_id"]
    sub = rbr_df[rbr_df["player_id"] == pid].sort_values(["year", "match_date"])
    if sub.empty:
        raise LookupError_(f"No rows for {who['player_name']}.")
    career_avg = sub[stat].mean(skipna=True)
    recent = sub.tail(int(recent_n))
    recent_avg = recent[stat].mean(skipna=True)
    return {
        "player_id": pid,
        "player_name": who["player_name"],
        "stat": stat,
        "recent_n": int(recent_n),
        "recent_avg": round(float(recent_avg), 2),
        "career_avg": round(float(career_avg), 2),
        "career_games": int(len(sub)),
        "delta": round(float(recent_avg - career_avg), 2),
    }


# ---------------------------------------------------------------------
# TOOL 6: structured lookup - top players for a team/season by a stat
# ---------------------------------------------------------------------
def get_team_top_players(team: str, season: int, stat: str = "fantasy_points", top_n: int = 5) -> dict:
    """Exact leaderboard of a team's players ranked by average `stat`
    across a given season, with real player names attached, computed
    from the raw round-by-round rows."""
    resolved = resolve_team(team)
    if stat not in ("disposals", "goals", "fantasy_points"):
        raise LookupError_(f"Unsupported stat '{stat}'.")
    sub = rbr_df[(rbr_df["team"] == resolved) & (rbr_df["year"] == int(season))]
    if sub.empty:
        raise LookupError_(f"No rows for {resolved} in season {season}.")
    grp = sub.groupby("player_id")[stat].mean().sort_values(ascending=False).head(int(top_n))
    leaders = []
    for pid, v in grp.items():
        name = info_df.loc[pid, "player_name"] if pid in info_df.index else f"Player {pid}"
        leaders.append({"player_id": int(pid), "player_name": name, f"avg_{stat}": round(float(v), 2)})
    return {"team": resolved, "season": int(season), "stat": stat, "leaders": leaders}


if __name__ == "__main__":
    print(resolve_player("Gary Ablett"))
    print(get_team_record_vs_opponent("Carlton Blues", "Collingwood Magpies"))
    print(get_team_recent_form("Carlton Blues", 3))
    print(get_team_top_players("Carlton Blues", 2024, "fantasy_points", 3))
