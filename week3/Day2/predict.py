"""
predict.py — callable inference functions for the AFL match-winner and top-player models.

Usage:
    from predict import predict_match_winner, predict_top_player

    predict_match_winner("Carlton Blues", "Collingwood Magpies", "2025-09-01")
    predict_top_player("Carlton Blues")
"""
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

_ARTIFACT_DIR = os.path.dirname(os.path.abspath(__file__))

_match_model = joblib.load(os.path.join(_ARTIFACT_DIR, "match_winner_model.joblib"))
_player_model = joblib.load(os.path.join(_ARTIFACT_DIR, "top_player_model.joblib"))
_reference = joblib.load(os.path.join(_ARTIFACT_DIR, "model_reference_data.joblib"))
_team_form = pd.read_csv(os.path.join(_ARTIFACT_DIR, "latest_team_form_snapshot.csv"), index_col="team")
_player_form = pd.read_csv(os.path.join(_ARTIFACT_DIR, "latest_player_form_snapshot.csv"))

_VALID_TEAMS = set(_reference["valid_teams"])
_VALID_VENUES = set(_reference["valid_venues"])
_DATA_MIN_DATE = datetime.fromisoformat(_reference["data_min_date"])
_DATA_MAX_DATE = datetime.fromisoformat(_reference["data_max_date"])
_MATCH_FEATURE_COLS = _reference["match_feature_cols"]
_PLAYER_FEATURE_COLS = _reference["player_feature_cols"]


class AFLModelInputError(ValueError):
    """Raised when predict_* is called with an input the models cannot handle."""


def _validate_team(team_name: str):
    if team_name not in _VALID_TEAMS:
        raise AFLModelInputError(
            f"Unknown team '{team_name}'. Valid teams: {sorted(_VALID_TEAMS)}"
        )


def _validate_date(date_str: str):
    try:
        d = pd.to_datetime(date_str)
    except Exception as e:
        raise AFLModelInputError(f"Could not parse date '{date_str}': {e}")
    if d < _DATA_MIN_DATE:
        raise AFLModelInputError(
            f"Date {date_str} is before the training data range ({_DATA_MIN_DATE.date()})."
        )
    # Note: dates after _DATA_MAX_DATE are allowed (that's the whole point -- predicting
    # a future match) but we warn implicitly by using the *latest known* team/player form,
    # which is the best available proxy for "current form" beyond the data window.
    return d


def predict_match_winner(team_a: str, team_b: str, date: str, venue: str = None) -> dict:
    """
    Predict the outcome of a match between team_a (treated as home) and team_b (away).

    Parameters
    ----------
    team_a : home team name (must match a canonical team name in the training data)
    team_b : away team name
    date   : ISO date string, e.g. "2025-09-01"
    venue  : optional venue name; if omitted or unknown, the model falls back gracefully
             (OneHotEncoder(handle_unknown="ignore") in the pipeline)

    Returns
    -------
    dict with keys: home_team, away_team, winner, probabilities {home, away, draw}
    """
    _validate_team(team_a)
    _validate_team(team_b)
    if team_a == team_b:
        raise AFLModelInputError("team_a and team_b must be different teams.")
    _validate_date(date)
    if venue is not None and venue not in _VALID_VENUES:
        venue = None  # let the encoder treat it as unknown rather than hard failing

    if team_a not in _team_form.index or team_b not in _team_form.index:
        raise AFLModelInputError("No recent form data available for one of the requested teams.")

    home_row = _team_form.loc[team_a]
    away_row = _team_form.loc[team_b]

    row = {
        "h2h_prior_meetings": np.nan,       # unknown for an arbitrary future pairing without a lookup table
        "h2h_home_team_winrate_prior": np.nan,
        "home_team_score_last5_avg": home_row.get("team_score_last5_avg", np.nan),
        "home_team_margin_last5_avg": home_row.get("team_margin_last5_avg", np.nan),
        "home_win_rate_last5_avg": home_row.get("win_rate_last5_avg", np.nan),
        "home_win_streak_entering": home_row.get("win_streak_entering", np.nan),
        "home_days_rest": home_row.get("days_rest", np.nan),
        "home_ladder_pos_entering": home_row.get("ladder_pos_entering", np.nan),
        "home_cum_percentage_entering": home_row.get("cum_percentage_entering", np.nan),
        "away_team_score_last5_avg": away_row.get("team_score_last5_avg", np.nan),
        "away_team_margin_last5_avg": away_row.get("team_margin_last5_avg", np.nan),
        "away_win_rate_last5_avg": away_row.get("win_rate_last5_avg", np.nan),
        "away_win_streak_entering": away_row.get("win_streak_entering", np.nan),
        "away_days_rest": away_row.get("days_rest", np.nan),
        "away_ladder_pos_entering": away_row.get("ladder_pos_entering", np.nan),
        "away_cum_percentage_entering": away_row.get("cum_percentage_entering", np.nan),
        "venue": venue,
    }
    row["diff_team_score_last5_avg"] = row["home_team_score_last5_avg"] - row["away_team_score_last5_avg"]
    row["diff_team_margin_last5_avg"] = row["home_team_margin_last5_avg"] - row["away_team_margin_last5_avg"]
    row["diff_win_rate_last5_avg"] = row["home_win_rate_last5_avg"] - row["away_win_rate_last5_avg"]
    row["diff_ladder_pos_entering"] = row["home_ladder_pos_entering"] - row["away_ladder_pos_entering"]
    row["diff_cum_percentage_entering"] = row["home_cum_percentage_entering"] - row["away_cum_percentage_entering"]
    row["diff_days_rest"] = row["home_days_rest"] - row["away_days_rest"]

    X = pd.DataFrame([row])[_MATCH_FEATURE_COLS]
    proba = _match_model.predict_proba(X)[0]
    classes = _match_model.classes_
    prob_dict = {cls: round(float(p), 3) for cls, p in zip(classes, proba)}
    winner = max(prob_dict, key=prob_dict.get)

    return {
        "home_team": team_a,
        "away_team": team_b,
        "winner": winner,
        "probabilities": prob_dict,
    }


def predict_top_player(team: str, stat_type: str = "fantasy_points", top_n: int = 5) -> list:
    """
    Rank a team's players by predicted output for their next match, using each
    player's most recent rolling-form snapshot.

    Parameters
    ----------
    team      : team name (must match a canonical team name in the training data)
    stat_type : currently only "fantasy_points" is modelled directly; kept as a
                parameter for interface stability with future stat-specific models
    top_n     : how many players to return

    Returns
    -------
    list of dicts: [{player_id, predicted_fantasy_points}, ...] sorted descending
    """
    _validate_team(team)
    if stat_type != "fantasy_points":
        raise AFLModelInputError(
            f"stat_type '{stat_type}' is not supported yet -- only 'fantasy_points' is modelled."
        )

    team_players = _player_form[_player_form["team"] == team].dropna(subset=["fantasy_points_last3_avg"])
    if team_players.empty:
        raise AFLModelInputError(f"No recent player data available for team '{team}'.")

    X = team_players[_PLAYER_FEATURE_COLS]
    preds = _player_model.predict(X)
    team_players = team_players.assign(predicted_fantasy_points=preds)
    ranked = team_players.sort_values("predicted_fantasy_points", ascending=False).head(top_n)

    return [
        {"player_id": int(r.player_id), "predicted_fantasy_points": round(float(r.predicted_fantasy_points), 1)}
        for r in ranked.itertuples()
    ]


if __name__ == "__main__":
    # quick manual smoke test
    print(predict_match_winner("Carlton Blues", "Collingwood Magpies", "2025-09-01"))
    print(predict_top_player("Carlton Blues"))
