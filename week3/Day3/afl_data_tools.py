from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
SEASON = DATA / 'afl_players_seasonal_stats_raw.csv'
ROUND = DATA / 'afl_players_round_by_round_stats_raw - afl_players_round_by_round_stats_raw.csv.csv'
TEAMS = DATA / 'team_matches_home_away_raw - team_matches_home_away_raw.csv.csv'
PLAYERS = DATA / 'afl_players_info_raw.csv'

_season = _round = _teams = _players = None

def _load():
    global _season, _round, _teams, _players
    if _season is None:
        _season = pd.read_csv(SEASON, low_memory=False)
        _season['player_id'] = pd.to_numeric(_season['player_id'], errors='coerce')
        _season['year'] = pd.to_numeric(_season['year'], errors='coerce')
        _round = pd.read_csv(ROUND, low_memory=False)
        _round['player_id'] = pd.to_numeric(_round['player_id'], errors='coerce')
        _round['year'] = pd.to_numeric(_round['year'], errors='coerce')
        _teams = pd.read_csv(TEAMS, low_memory=False)
        _players = pd.read_csv(PLAYERS, low_memory=False)
        _players['id'] = pd.to_numeric(_players['id'], errors='coerce')
        _teams['year'] = pd.to_numeric(_teams['year'], errors='coerce')
    return _season, _round, _teams, _players

def _player_id(value):
    _, _, _, players = _load()
    s = str(value).strip()
    if s.isdigit(): return int(s)
    hits = players[players['player_name'].astype(str).str.contains(s, case=False, na=False, regex=False)].drop_duplicates(subset=['id'])
    if hits.empty:
        raise ValueError(f'Player not found in the dataset: {value}')
    if len(hits) > 1:
        names = ', '.join(hits['player_name'].drop_duplicates().head(8))
        raise ValueError(f"Player name is ambiguous. Matches: {names}")
    return int(hits.iloc[0]['id'])

def _player_name(pid):
    _, _, _, players = _load()
    h = players[players['id'].astype(str) == str(pid)]
    return h.iloc[0]['player_name'] if not h.empty else str(pid)

def get_player_season_stats(player, season):
    """Look up a player's exact season totals and per-game averages from the AFL seasonal dataset."""
    s, _, _, _ = _load(); pid = _player_id(player); season = int(season)
    r = s[(s.player_id == pid) & (s.year == season)]
    if r.empty: return {'found': False, 'player': _player_name(pid), 'season': season, 'source': SEASON.name}
    row = r.iloc[0]
    fields = ['games_played','disposals','goals','marks','kicks','handballs','tackles','clearances','inside_50s','total_fantasy_points','avg_disposals','avg_goals','avg_marks','avg_tackles','avg_fantasy_points']
    return {'found': True, 'player': _player_name(pid), 'player_id': pid, 'season': season,
            'stats': {f: None if pd.isna(row[f]) else float(row[f]) for f in fields}, 'source': SEASON.name}

def get_player_recent_rounds(player, n=1, before_season=None, before_round=None):
    """Return the most recent n match rows for a player, optionally immediately before a given season/round."""
    _, r, _, _ = _load(); pid = _player_id(player); n = max(1, int(n)); x = r[r.player_id == pid].copy()
    x['match_date'] = pd.to_datetime(x['match_date'], errors='coerce')
    x = x.sort_values(['match_date','year'], ascending=False)
    if before_season is not None:
        bs = int(before_season)
        if before_round is not None:
            br = int(before_round)
            x = x[(x.year < bs) | ((x.year == bs) & (pd.to_numeric(x['round'], errors='coerce') < br))]
        else: x = x[x.year <= bs]
    cols = ['year','round','match_date','team','opponent','result','disposals','kicks','handballs','marks','goals','tackles','clearances','fantasy_points']
    rows=[]
    for _, q in x.head(n).iterrows():
        d={c:(None if pd.isna(q[c]) else q[c]) for c in cols}; d['match_date']=str(d['match_date'].date()) if d['match_date'] else None
        rows.append(d)
    return {'found': bool(rows), 'player': _player_name(pid), 'player_id': pid, 'games': rows, 'source': ROUND.name}

def compare_player_recent_to_career(player, stat='disposals', recent_n=5):
    """Compare a player's recent average for a stat with the exact career average computed from match-level AFL data."""
    _, r, _, _ = _load(); pid = _player_id(player); stat = str(stat).strip().lower(); recent_n=max(1,int(recent_n))
    allowed={'disposals','kicks','marks','handballs','goals','tackles','clearances','inside_50s','fantasy_points'}
    if stat not in allowed: raise ValueError(f'Stat must be one of: {sorted(allowed)}')
    x=r[r.player_id==pid].copy(); x['match_date']=pd.to_datetime(x['match_date'], errors='coerce'); x[stat]=pd.to_numeric(x[stat],errors='coerce'); x=x.dropna(subset=[stat]).sort_values('match_date',ascending=False)
    recent=x.head(recent_n)[stat].mean(); career=x[stat].mean()
    return {'found': len(x)>0, 'player': _player_name(pid), 'stat': stat, 'recent_games_used': min(recent_n,len(x)), 'recent_average': None if pd.isna(recent) else round(float(recent),2), 'career_average': None if pd.isna(career) else round(float(career),2), 'source': ROUND.name}

def get_team_record_vs_opponent(team, opponent):
    """Compute exact historical wins, losses, draws and meetings for one AFL team against another from match results."""
    _, _, t, _ = _load(); a=str(team).strip(); b=str(opponent).strip()
    x=t[(t.team_name.str.casefold()==a.casefold()) & (t.opponent.str.casefold()==b.casefold())]
    if x.empty:
        # tolerate partial team names
        x=t[t.team_name.str.contains(a,case=False,na=False,regex=False) & t.opponent.str.contains(b,case=False,na=False,regex=False)]
    counts=x.result.astype(str).str.upper().value_counts()
    return {'found': not x.empty, 'team': a, 'opponent': b, 'meetings': int(len(x)), 'wins': int(counts.get('W',0)), 'losses': int(counts.get('L',0)), 'draws': int(counts.get('D',0)), 'source': TEAMS.name}

def player_search(name):
    """Resolve an AFL player name to the dataset player ID; useful before exact stat lookups."""
    _load(); s=str(name).strip(); h=_players[_players.player_name.astype(str).str.contains(s,case=False,na=False,regex=False)][['id','player_name','player_teams']].head(10)
    return h.to_dict('records')
