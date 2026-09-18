"""
afl_predict_tools.py
----------------------
Day 4 / Task 3: prediction tools (match winner + top player), trained
directly from the real datasets already loaded in afl_data_tools.py
(team_matches_home_away_raw, afl_players_round_by_round_stats_raw).

Both models are trained once at import time (a few seconds) so this
module is fully self-contained - no external .joblib artifacts needed.
This intentionally mirrors the Day 2 modelling approach (rolling-form
features, no leakage via shift(1), GradientBoosting) but is re-fit on
the fuller raw history rather than the Day-2 snapshot files, and is
kept here specifically so it can be wrapped as LangGraph tool nodes.
"""

import re
import numpy as np
import pandas as pd
from datetime import date
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

import afl_data_tools as dt   # reuses the already-cleaned team_matches_df / rbr_df / ALL_TEAMS

TODAY = date.today()  # "now", for resolving relative dates like "this week"


class PredictionInputError(ValueError):
    """Raised when a prediction request can't be resolved (bad team,
    unsupported stat, no recent form) - the graph's validation node
    catches this and routes to clarification instead of guessing."""


# ---------------------------------------------------------------- aliases --
NICKNAME_MAP = {
    "pies": "Collingwood Magpies", "magpies": "Collingwood Magpies",
    "cats": "Geelong Cats", "blues": "Carlton Blues",
    "bombers": "Essendon Bombers", "dons": "Essendon Bombers",
    "tigers": "Richmond Tigers", "hawks": "Hawthorn Hawks",
    "demons": "Melbourne Demons", "dees": "Melbourne Demons",
    "swans": "Sydney Swans", "eagles": "West Coast Eagles",
    "dockers": "Fremantle Dockers", "freo": "Fremantle Dockers",
    "lions": "Brisbane Lions", "crows": "Adelaide Crows",
    "power": "Port Adelaide Power", "saints": "St Kilda Saints",
    "roos": "North Melbourne Kangaroos", "kangaroos": "North Melbourne Kangaroos",
    "bulldogs": "Western Bulldogs", "dogs": "Western Bulldogs",
    "giants": "Greater Western Sydney Giants", "gws": "Greater Western Sydney Giants",
    "suns": "Gold Coast Suns",
}


def resolve_team_alias(name: str) -> str:
    """Resolve a nickname ('Pies', 'Cats', 'the Dogs') or a full/partial
    team name to the dataset's canonical team key. Raises
    PredictionInputError (not a silent guess) if nothing matches well
    enough - the graph's validation node uses this to trigger a
    clarification loop rather than predicting for the wrong team."""
    if not name:
        raise PredictionInputError("No team name given.")
    key = re.sub(r"^\s*(the)\s+", "", name.strip().lower())
    if key in NICKNAME_MAP:
        return NICKNAME_MAP[key]
    try:
        return dt.resolve_team(name)
    except dt.LookupError_:
        raise PredictionInputError(f"Could not resolve '{name}' to a known AFL team.")


def resolve_relative_date(text: str) -> str:
    """Resolve a relative time phrase ('this week', 'this round', 'next
    round', 'today') to an ISO date. LIMITATION (documented, not hidden):
    the dataset has no forward-looking fixture list, so 'this week' /
    'next round' cannot be mapped to a real scheduled fixture - we fall
    back to today's real-world date, which is enough for the model
    (it only needs a valid date >= the training window) but the agent
    must be honest that it isn't reading an actual future fixture."""
    t = text.lower()
    if re.search(r"\btoday\b|\bthis week\b|\bthis round\b|\bnext round\b|\bnext week\b|\bupcoming\b", t):
        return TODAY.isoformat()
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", t)
    if m:
        return m.group(1)
    return TODAY.isoformat()


# --------------------------------------------------- feature engineering --
def _team_rolling_features(window: int = 5) -> pd.DataFrame:
    """One row per (team, match_date): rolling averages of that team's
    OWN score/margin/win-rate over their previous `window` matches,
    computed with shift(1) so the row reflects form *entering* that
    match (no leakage of the match's own result into its own features).
    Also includes days_rest (gap since the team's previous match)."""
    df = dt.team_matches_df.copy().sort_values(["team_name", "match_date"])
    df["win_flag"] = (df["result"] == "W").astype(float)
    g = df.groupby("team_name")
    df["score_last_avg"] = g["team_score"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["margin_last_avg"] = g["margin"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["winrate_last_avg"] = g["win_flag"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    df["days_rest"] = g["match_date"].transform(lambda s: (s - s.shift(1)).dt.days)
    return df[["team_name", "match_date", "home_away", "opponent", "result",
               "score_last_avg", "margin_last_avg", "winrate_last_avg", "days_rest"]]


def _build_match_training_frame():
    feats = _team_rolling_features()
    home = feats[feats["home_away"] == "H"].copy()
    away_lookup = feats.set_index(["team_name", "match_date"])[
        ["score_last_avg", "margin_last_avg", "winrate_last_avg", "days_rest"]
    ]
    rows = []
    for r in home.itertuples(index=False):
        try:
            away_feat = away_lookup.loc[(r.opponent, r.match_date)]
        except KeyError:
            continue
        rows.append({
            "home_score_last_avg": r.score_last_avg, "away_score_last_avg": away_feat["score_last_avg"],
            "home_margin_last_avg": r.margin_last_avg, "away_margin_last_avg": away_feat["margin_last_avg"],
            "home_winrate_last_avg": r.winrate_last_avg, "away_winrate_last_avg": away_feat["winrate_last_avg"],
            "home_days_rest": r.days_rest, "away_days_rest": away_feat["days_rest"],
            "diff_score": r.score_last_avg - away_feat["score_last_avg"],
            "diff_margin": r.margin_last_avg - away_feat["margin_last_avg"],
            "diff_winrate": r.winrate_last_avg - away_feat["winrate_last_avg"],
            "result": r.result,
        })
    out = pd.DataFrame(rows).dropna()
    return out


_MATCH_FEATURE_COLS = ["home_score_last_avg", "away_score_last_avg", "home_margin_last_avg",
                        "away_margin_last_avg", "home_winrate_last_avg", "away_winrate_last_avg",
                        "home_days_rest", "away_days_rest", "diff_score", "diff_margin", "diff_winrate"]

_match_train_df = _build_match_training_frame()
_match_model = GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=2)
_match_model.fit(_match_train_df[_MATCH_FEATURE_COLS], _match_train_df["result"])
_MATCH_FEATURE_IMPORTANCE = sorted(
    zip(_MATCH_FEATURE_COLS, _match_model.feature_importances_), key=lambda x: -x[1]
)

# "current form" snapshot per team = their most recent match's rolling
# window INCLUDING that match (used only at inference time, for a future/
# hypothetical match - never used as a training label's own features)
_current_form = dt.team_matches_df.copy().sort_values(["team_name", "match_date"])
_current_form["win_flag"] = (_current_form["result"] == "W").astype(float)
g = _current_form.groupby("team_name")
_current_form["score_last_avg"] = g["team_score"].transform(lambda s: s.rolling(5, min_periods=1).mean())
_current_form["margin_last_avg"] = g["margin"].transform(lambda s: s.rolling(5, min_periods=1).mean())
_current_form["winrate_last_avg"] = g["win_flag"].transform(lambda s: s.rolling(5, min_periods=1).mean())
_current_form["days_rest"] = g["match_date"].transform(lambda s: (s - s.shift(1)).dt.days)
_TEAM_CURRENT_FORM = _current_form.sort_values("match_date").groupby("team_name").tail(1).set_index("team_name")


def predict_match_winner(team_a: str, team_b: str, date_str: str = None) -> dict:
    """Predict the winner of team_a (home) vs team_b (away) using each
    team's current rolling form, plus a top-3 feature explanation.
    Returns probabilities for every class the model saw (W/L/D)."""
    a = resolve_team_alias(team_a)
    b = resolve_team_alias(team_b)
    if a == b:
        raise PredictionInputError("team_a and team_b must be different teams.")
    if a not in _TEAM_CURRENT_FORM.index or b not in _TEAM_CURRENT_FORM.index:
        raise PredictionInputError(f"No recent form data available for '{team_a}' or '{team_b}'.")
    ha, hb = _TEAM_CURRENT_FORM.loc[a], _TEAM_CURRENT_FORM.loc[b]
    row = {
        "home_score_last_avg": ha["score_last_avg"], "away_score_last_avg": hb["score_last_avg"],
        "home_margin_last_avg": ha["margin_last_avg"], "away_margin_last_avg": hb["margin_last_avg"],
        "home_winrate_last_avg": ha["winrate_last_avg"], "away_winrate_last_avg": hb["winrate_last_avg"],
        "home_days_rest": ha["days_rest"] if pd.notna(ha["days_rest"]) else 7,
        "away_days_rest": hb["days_rest"] if pd.notna(hb["days_rest"]) else 7,
    }
    row["diff_score"] = row["home_score_last_avg"] - row["away_score_last_avg"]
    row["diff_margin"] = row["home_margin_last_avg"] - row["away_margin_last_avg"]
    row["diff_winrate"] = row["home_winrate_last_avg"] - row["away_winrate_last_avg"]
    X = pd.DataFrame([row])[_MATCH_FEATURE_COLS]
    proba = _match_model.predict_proba(X)[0]
    prob_dict = {cls: round(float(p), 3) for cls, p in zip(_match_model.classes_, proba)}
    winner = max(prob_dict, key=prob_dict.get)
    top_features = [f"{name} ({'home' if row[name]>=0 else 'away'} favoured, value={row[name]:.1f})"
                    for name, _ in _MATCH_FEATURE_IMPORTANCE[:3]]
    return {
        "home_team": a, "away_team": b, "predicted_winner": winner,
        "probabilities": prob_dict,
        "top_features": top_features,
        "disclaimer": "This is a probabilistic model estimate, not a certain outcome.",
    }


# ------------------------------------------------------- top-player model --
def _build_player_training_frame():
    df = dt.rbr_df.copy().sort_values(["player_id", "match_date"])
    g = df.groupby("player_id")
    df["fp_last3_avg"] = g["fantasy_points"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["fp_last5_avg"] = g["fantasy_points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    df["disp_last5_avg"] = g["disposals"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    out = df.dropna(subset=["fp_last3_avg", "fp_last5_avg", "disp_last5_avg", "fantasy_points"])
    return out[["fp_last3_avg", "fp_last5_avg", "disp_last5_avg", "fantasy_points", "player_id", "match_date", "team"]]


_PLAYER_FEATURE_COLS = ["fp_last3_avg", "fp_last5_avg", "disp_last5_avg"]
_player_train_df = _build_player_training_frame()
_player_model = GradientBoostingRegressor(random_state=42, n_estimators=150, max_depth=2)
_player_model.fit(_player_train_df[_PLAYER_FEATURE_COLS], _player_train_df["fantasy_points"])
_PLAYER_FEATURE_IMPORTANCE = sorted(
    zip(_PLAYER_FEATURE_COLS, _player_model.feature_importances_), key=lambda x: -x[1]
)

# current per-player rolling form (not shifted - includes latest match)
_pcur = dt.rbr_df.copy().sort_values(["player_id", "match_date"])
gp = _pcur.groupby("player_id")
_pcur["fp_last3_avg"] = gp["fantasy_points"].transform(lambda s: s.rolling(3, min_periods=1).mean())
_pcur["fp_last5_avg"] = gp["fantasy_points"].transform(lambda s: s.rolling(5, min_periods=1).mean())
_pcur["disp_last5_avg"] = gp["disposals"].transform(lambda s: s.rolling(5, min_periods=1).mean())
_PLAYER_CURRENT_FORM = _pcur.sort_values("match_date").groupby("player_id").tail(1)


def predict_top_player(team: str, stat_type: str = "fantasy_points", top_n: int = 5) -> dict:
    """Rank a team's players by predicted fantasy points for a
    hypothetical next match, using each player's current rolling form.
    Only 'fantasy_points' is modelled - any other stat_type is a clean
    'not supported' error, not a silent wrong answer."""
    a = resolve_team_alias(team)
    if stat_type != "fantasy_points":
        raise PredictionInputError(f"stat_type '{stat_type}' is not modelled yet - only 'fantasy_points' is.")
    squad = _PLAYER_CURRENT_FORM[_PLAYER_CURRENT_FORM["team"] == a].dropna(subset=_PLAYER_FEATURE_COLS)
    if squad.empty:
        raise PredictionInputError(f"No recent player form data available for '{a}'.")
    X = squad[_PLAYER_FEATURE_COLS]
    preds = _player_model.predict(X)
    squad = squad.assign(predicted_fantasy_points=preds)
    ranked = squad.sort_values("predicted_fantasy_points", ascending=False).head(int(top_n))
    leaders = []
    for r in ranked.itertuples():
        name = dt.info_df.loc[r.player_id, "player_name"] if r.player_id in dt.info_df.index else f"Player {r.player_id}"
        leaders.append({"player_id": int(r.player_id), "player_name": name,
                         "predicted_fantasy_points": round(float(r.predicted_fantasy_points), 1)})
    top_features = [name for name, _ in _PLAYER_FEATURE_IMPORTANCE[:2]]
    return {
        "team": a, "stat_type": stat_type, "leaders": leaders,
        "top_features": top_features,
        "disclaimer": "This is a probabilistic model estimate, not a certain outcome.",
    }


if __name__ == "__main__":
    print(predict_match_winner("Pies", "Cats", "this week"))
    print(predict_top_player("Carlton Blues"))
