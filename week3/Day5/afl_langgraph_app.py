"""Day 5 capstone: hardened LangGraph AFL assistant.
Real structured retrieval/prediction tools operate on the supplied AFL CSVs.
Gemini is used only to phrase a response after deterministic routing/tool execution.
If GEMINI_API_KEY is absent or quota is exhausted, deterministic formatting keeps the app runnable.
"""
import os, re, time, json
from typing import TypedDict, Any, Dict, List, Tuple, Optional
from afl_guardrails import classify_scope, refusal_response, SYSTEM_PROMPT
import afl_data_tools as dt
import afl_predict_tools as pred

PRED_PAT = re.compile(r"\b(who will win|who'll win|predict|prediction|probability|chances|odds|forecast|will .* win|will .* beat|top.?scor\w*|top player|best player|predicted)\b", re.I)

class State(TypedDict, total=False):
    user_query: str; conversation_id: str; history: List[Tuple[str,str]]; memory: Dict[str,Any]
    intent: str; tool_name: Optional[str]; tool_result: Optional[dict]; tool_error: Optional[str]
    final_response: str; prediction_metadata: Optional[dict]; trace: List[dict]; latency_ms: float

class RateLimiter:
    def __init__(self, limit=12, window=60): self.limit,self.window=limit,window; self.hits={}
    def allow(self, cid):
        now=time.time(); arr=[x for x in self.hits.get(cid,[]) if now-x<self.window]
        if len(arr)>=self.limit: self.hits[cid]=arr; return False
        arr.append(now); self.hits[cid]=arr; return True
rate_limiter=RateLimiter()
MEMORY: Dict[str,Dict[str,Any]]={}; HISTORY: Dict[str,List[Tuple[str,str]]]={}

def _teams(text):
    """Extract teams in the exact order they appear in the user's text.
    Never invent a second team; if only one unique team is named, callers
    must ask for the opponent instead of silently substituting one.

    Fix applied: the original version only recognised full canonical names
    (e.g. "Carlton Blues") and fancy nicknames (e.g. "Blues", "Pies") but NOT
    the plain first-word/city name most people actually type -- "Carlton vs
    Collingwood" matched neither list and failed with "I need two different
    AFL teams", even though both teams were clearly named. Common first-word
    aliases are added below, with explicit disambiguation for words that are
    ambiguous across two real teams (e.g. "Brisbane" -> Bears vs Lions,
    "Melbourne" alone vs "North Melbourne").

    Also fixed: overlap suppression. A short alias contained inside a longer
    match (e.g. bare "melbourne" sitting inside "north melbourne") must not
    register as a second, spurious team hit at the same text position -- the
    longest match at a given span always wins.
    """
    tl=text.lower()
    nick={'pies':'Collingwood Magpies','cats':'Geelong Cats','blues':'Carlton Blues','bombers':'Essendon Bombers','dons':'Essendon Bombers','tigers':'Richmond Tigers','hawks':'Hawthorn Hawks','dees':'Melbourne Demons','demons':'Melbourne Demons','swans':'Sydney Swans','eagles':'West Coast Eagles','dockers':'Fremantle Dockers','freo':'Fremantle Dockers','lions':'Brisbane Lions','crows':'Adelaide Crows','power':'Port Adelaide Power','saints':'St Kilda Saints','roos':'North Melbourne Kangaroos','kangaroos':'North Melbourne Kangaroos','bulldogs':'Western Bulldogs','dogs':'Western Bulldogs','giants':'Greater Western Sydney Giants','gws':'Greater Western Sydney Giants','suns':'Gold Coast Suns',
        'carlton':'Carlton Blues','collingwood':'Collingwood Magpies','essendon':'Essendon Bombers',
        'geelong':'Geelong Cats','fremantle':'Fremantle Dockers','richmond':'Richmond Tigers',
        'hawthorn':'Hawthorn Hawks','adelaide':'Adelaide Crows','sydney':'Sydney Swans',
        'fitzroy':'Fitzroy Lions','western bulldogs':'Western Bulldogs','footscray':'Western Bulldogs',
        'port adelaide':'Port Adelaide Power','west coast':'West Coast Eagles',
        'gold coast':'Gold Coast Suns','st kilda':'St Kilda Saints',
        'north melbourne':'North Melbourne Kangaroos','melbourne':'Melbourne Demons',
        'brisbane lions':'Brisbane Lions','brisbane bears':'Brisbane Bears','brisbane':'Brisbane Lions',
        'greater western sydney':'Greater Western Sydney Giants'}
    raw=[]
    for team in dt.ALL_TEAMS:
        for m in re.finditer(r'(?<![a-z])'+re.escape(team.lower())+r'(?![a-z])',tl):
            raw.append((m.start(),m.end(),team))
    for k,v in nick.items():
        for m in re.finditer(r'(?<![a-z])'+re.escape(k)+r'(?![a-z])',tl):
            raw.append((m.start(),m.end(),v))
    raw.sort(key=lambda x:(-(x[1]-x[0]), x[0]))
    accepted=[]
    for s,e,team in raw:
        if any(not(e<=as_ or s>=ae) for as_,ae,_ in accepted):
            continue
        accepted.append((s,e,team))
    accepted.sort(key=lambda x:x[0])
    out=[]
    for _,_,team in accepted:
        if team not in out: out.append(team)
    return out[:3]

_LASTNAME_TO_IDS = {}
for _pid, _row in dt.info_df.iterrows():
    _ln = str(_row.get('last_name', '')).strip().lower()
    if _ln:
        _LASTNAME_TO_IDS.setdefault(_ln, []).append(_pid)

def _player(text,mem):
    """Resolve a player mentioned in the text to a player_id.

    Fix applied: the original version only matched if the user typed the
    FULL name exactly as stored (e.g. "Marcus Bontempelli"), because it
    checked `if full_name in text.lower()`. Real usage very often uses just
    a surname ("Bontempelli's stats", "how did Walsh go") -- that never
    matched at all and silently fell through to "I couldn't complete that
    lookup", even though the player is unambiguous. Now also matches on
    last_name for any capitalised word in the query, preferring a player
    already in conversational memory when a surname is shared by more than
    one player (a real but rare ambiguity in a >2,900-player dataset).
    """
    tl = text.lower()
    for nm,pid in dt.NAME_TO_ID.items():
        if nm in tl: return dt.resolve_player(pid)
    for word in re.findall(r"\b([A-Z][a-zA-Z]+)(?:'s)?\b", text):
        ids = _LASTNAME_TO_IDS.get(word.lower())
        if ids:
            if mem.get('last_player_id') in ids:
                return dt.resolve_player(mem['last_player_id'])
            return dt.resolve_player(ids[0])
    m=re.search(r'(?:player[_ ]?(?:id)?|pid)\s*#?\s*(\d{3,6})',text,re.I)
    if m:
        try:return dt.resolve_player(int(m.group(1)))
        except: pass
    if mem.get('last_player_id'): return dt.resolve_player(mem['last_player_id'])
    return None

def router_node(st):
    scope=classify_scope(st['user_query'], [x for r,x in st.get('history',[])[-6:] if r=='user'])
    st['trace'].append({'node':'router','scope':scope})
    if not scope['in_scope']: st['intent']='off_topic'
    else: st['intent']='prediction' if PRED_PAT.search(st['user_query']) else 'retrieval'
    return st

def retrieval_node(st):
    q=st['user_query']; t=q.lower(); m=st['memory'];
    try:
        if re.search(r'head.to.head|record (vs|against)|played each other',t):
            ts=_teams(q); 
            if len(ts)<2: raise dt.LookupError_('need_two_teams')
            r=dt.get_team_record_vs_opponent(*ts); st['tool_name']='get_team_record_vs_opponent'; m.update(last_team=ts[0],last_opponent=ts[1])
        elif 'career average' in t or 'compare' in t and 'career' in t:
            w=_player(q,m); 
            if not w: raise dt.LookupError_('need_player')
            stat='goals' if 'goal' in t else ('fantasy_points' if 'fantasy' in t else 'disposals')
            r=dt.compare_player_recent_to_career(w['player_id'],stat,5); st['tool_name']='compare_player_recent_to_career'; m.update(last_player_id=w['player_id'])
        elif re.search(r'last round|recent round|previous round|round before|game before',t):
            w=_player(q,m); 
            if not w: raise dt.LookupError_('need_player')
            before=None
            if m.get('last_games'):
                g=m['last_games'][0]; before=(g['year'],g['round'])
            r=dt.get_player_recent_rounds(w['player_id'],1,*(before or (None,None))); st['tool_name']='get_player_recent_rounds'; m.update(last_player_id=w['player_id'],last_games=r['games'])
        elif re.search(r'\b(which|who|best|strongest|top) player\b|strongest recent form|best form',t) and 'will' not in t and 'predict' not in t:
            # Fix: this phrasing (retrieval framing -- "which player had the
            # strongest recent form", NOT "who WILL top-score", which is a
            # prediction) had no matching branch at all -- it fell through to
            # the generic catch-all, found no team or player entity in the
            # sentence itself, and returned "I couldn't complete that lookup"
            # even right after the user had just been talking about a team.
            # This exact phrasing is literally in this project's own
            # demo_script.md as a suggested live-demo line, so it needed to work.
            ts=_teams(q); team = ts[0] if ts else m.get('last_team')
            if not team: raise dt.LookupError_('need_team')
            season = m.get('last_season') or 2025
            r=dt.get_team_top_players(team, season); st['tool_name']='get_team_top_players'
            m.update(last_team=team, last_season=season)
            if r.get('leaders'): m['last_player_id']=r['leaders'][0]['player_id']
        elif re.search(r'season stats|season\b|\b(19|20)\d{2}\b',t):
            # Fix: originally only matched "stats IN 2023" -- real phrasing like
            # "stats FOR 2023" or "his 2023 season" never matched this branch at
            # all and silently fell through to the generic "recent rounds"
            # fallback below, returning the wrong data (most recent match
            # instead of the requested season's totals). Broadened to match
            # any 4-digit year, checked before the generic fallback.
            w=_player(q,m); y=re.search(r'\b(19|20)\d{2}\b',q); y=int(y.group()) if y else m.get('last_season')
            if not w or not y: raise dt.LookupError_('need_player_and_season')
            r=dt.get_player_season_stats(w['player_id'],y); st['tool_name']='get_player_season_stats'; m.update(last_player_id=w['player_id'],last_season=y)
        else:
            ts=_teams(q)
            if ts: r=dt.get_team_recent_form(ts[0],5); st['tool_name']='get_team_recent_form'; m['last_team']=ts[0]
            else:
                w=_player(q,m)
                if not w: raise dt.LookupError_('need_player_or_team')
                r=dt.get_player_recent_rounds(w['player_id'],5); st['tool_name']='get_player_recent_rounds'; m.update(last_player_id=w['player_id'],last_games=r['games'])
        st['tool_result']=r; st['tool_error']=None; st['trace'].append({'node':'retrieval','tool':st['tool_name'],'tool_result':r})
    except Exception as e: st['tool_error']=str(e); st['tool_result']=None; st['trace'].append({'node':'retrieval','error':str(e)})
    return st

TOP_PLAYER_PAT = re.compile(r"\btop.?scor|top player|best player|who will (get the most|kick the most|lead)\b", re.I)
STAT_WORD_PAT = re.compile(r"\b(tackles?|disposals?|marks?|kicks?|handballs?|clearances?|goals?)\b", re.I)

def prediction_node(st):
    q=st['user_query']; ts=_teams(q); m=st['memory']
    try:
        # Top-player / top-scorer prediction: single-team request, routed to
        # predict_top_player -- this was previously NEVER called anywhere in
        # this file (a real gap against Task 3, which explicitly requires
        # both Day 2 functions to be wired in). A stat word other than
        # fantasy points (e.g. "tackles") is passed through so the model's
        # own clean "not modelled yet" error surfaces, instead of the
        # generic (and wrong, for this case) "need two teams" message.
        if TOP_PLAYER_PAT.search(q) or (len(ts) <= 1 and STAT_WORD_PAT.search(q) and 'top' in q.lower()):
            team = ts[0] if ts else m.get('last_team')
            if not team:
                raise pred.PredictionInputError('Please tell me which team you want a top-player prediction for.')
            stat_m = STAT_WORD_PAT.search(q)
            stat_type = 'fantasy_points'
            if stat_m:
                word = stat_m.group(1).lower().rstrip('s')
                if word not in ('fantasy point', 'fantasy_point'):
                    stat_type = word + 's' if not word.endswith('s') else word
                    stat_type = {'tackle':'tackles','disposal':'disposals','mark':'marks','kick':'kicks',
                                 'handball':'handballs','clearance':'clearances','goal':'goals'}.get(word, stat_type)
            r = pred.predict_top_player(team, stat_type=stat_type)
            st['tool_name']='predict_top_player'; st['tool_result']=r; st['prediction_metadata']=r; st['tool_error']=None
            m['last_team']=team
            st['trace'].append({'node':'prediction','tool':st['tool_name'],'tool_result':r})
            return st

        if len(ts) >= 2 and ts[0] == ts[1]:
            raise pred.PredictionInputError('The same team was named twice. Please provide two different AFL teams.')
        # Only use conversational memory when the current message does not
        # name a team. A one-team request needs clarification, not a guessed
        # opponent.
        if len(ts) == 0:
            for x in [m.get('last_team'),m.get('last_opponent')]:
                if x and x not in ts: ts.append(x)
        if len(ts)<2: raise pred.PredictionInputError('Please provide two different AFL teams for the prediction.')
        r=pred.predict_match_winner(ts[0],ts[1]); st['tool_name']='predict_match_winner'; st['tool_result']=r; st['prediction_metadata']=r; st['tool_error']=None
        st['trace'].append({'node':'prediction','tool':st['tool_name'],'tool_result':r})
    except Exception as e: st['tool_error']=str(e); st['tool_result']=None; st['trace'].append({'node':'prediction','error':str(e)})
    return st

def format_deterministic(st):
    if st['intent']=='off_topic': return refusal_response(classify_scope(st['user_query'], [x for r,x in st.get('history',[]) if r=='user']))
    if st.get('tool_error'):
        err = st['tool_error']
        if st.get('intent') == 'prediction':
            # surface the model's OWN error message (e.g. "stat_type 'tackles'
            # is not modelled yet") instead of a generic, sometimes-wrong
            # "need two teams" message for every prediction failure
            if 'not modelled' in err.lower() or 'no recent player form' in err.lower() or 'no recent form data' in err.lower():
                return f"{err} This is a predicted probability, not a certainty."
            return "I need two different AFL teams for a prediction. Please provide both teams. This is a predicted probability, not a certainty."
        return "I couldn't complete that lookup from the AFL dataset. Please provide the missing team, player, or season detail; I won't guess a statistic."
    r=st.get('tool_result') or {}
    if st['intent']=='prediction':
        if 'leaders' in r:
            # top-player prediction response (Task 3: grounding features required)
            names = '; '.join(f"{p['player_name']} ({p['predicted_fantasy_points']} pts)" for p in r['leaders'])
            feats = ', '.join(r.get('top_features', []))
            return (f"Predicted top performer(s) for {r['team']} ({r['stat_type']}): {names}. "
                    f"Driven mainly by: {feats}. {r.get('disclaimer','This is a predicted probability, not a certainty.')}")
        # match-winner prediction response: name the actual favoured team, not a bare W/L/D letter
        p=r.get('probabilities',{})
        home, away = r.get('home_team'), r.get('away_team')
        winner_label = r.get('predicted_winner')
        winner_name = home if winner_label == 'W' else (away if winner_label == 'L' else 'a draw')
        probs = ', '.join(f'{k}: {v:.1%}' for k,v in p.items())
        feats = '; '.join(r.get('top_features', [])[:3])
        return (f"Model prediction: {home} vs {away} — {winner_name} favoured. "
                f"Probabilities (from {home}'s perspective, W/L/D): {probs}. "
                f"Driven mainly by: {feats}. This is a predicted probability, not a certainty.")
    if 'games_played' in r and 'avg_disposals' in r: return f"{r['player_name']} in {r['season']}: {r['games_played']} games, {r['total_disposals']:.0f} total disposals, {r['avg_disposals']:.2f} disposals/game, and {r['avg_fantasy_points']:.2f} fantasy points/game."
    if 'wins_for_team' in r: return f"{r['team']} vs {r['opponent']}: {r['games_played']} games — {r['wins_for_team']} wins, {r['losses_for_team']} losses, {r['draws']} draws."
    if 'recent_avg' in r and 'career_avg' in r:
        # compare_player_recent_to_career -- was previously falling through to raw JSON
        return (f"{r['player_name']}'s last {r['recent_n']}-game average {r['stat']}: {r['recent_avg']}, "
                f"vs a career average of {r['career_avg']} across {r['career_games']} games "
                f"({'+' if r['delta']>=0 else ''}{r['delta']} recent vs career).")
    if 'games' in r and 'player_name' in r and 'team' not in r:
        # get_player_recent_rounds -- was previously falling through to raw JSON
        g = r['games'][-1] if r['games'] else None
        if g is None: return f"No recent games found for {r['player_name']}."
        return (f"{r['player_name']}'s most recent match: {g['year']} Round {g['round']} vs {g['opponent']} — "
                f"{g['disposals']} disposals, {g['goals']} goals, {g['fantasy_points']} fantasy points ({g['result']}).")
    if 'games' in r and 'team' in r: return f"{r['team']} won {r.get('win_rate_pct',0):.1f}% of its last {r['n_returned']} recorded matches."
    if 'leaders' in r and 'season' in r:
        # get_team_top_players (retrieval: actual historical leaderboard) --
        # distinct from predict_top_player's 'leaders' shape (which has no
        # 'season' key and is handled in the prediction branch above)
        names = '; '.join(f"{p['player_name']} ({p['avg_fantasy_points']} avg {r['stat']})" for p in r['leaders'])
        return f"{r['team']}'s top players by {r['stat']} in {r['season']}: {names}."
    return json.dumps(r, default=str)

def gemini_phrase(st):
    key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not key: return format_deterministic(st)
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm=ChatGoogleGenerativeAI(model=os.getenv('GEMINI_MODEL','gemini-3.6-flash'), temperature=0, google_api_key=key)
        prompt=(SYSTEM_PROMPT+'\nUse ONLY this verified tool result; do not change any numeric value. '+
                'For predictions preserve the exact disclaimer: "predicted probability, not a certainty".\n\n'
                'User: '+st['user_query']+'\nVerified tool result: '+json.dumps(st.get('tool_result'),default=str))
        return llm.invoke(prompt).content
    except Exception:
        return format_deterministic(st)

def build_graph():
    try:
        from langgraph.graph import StateGraph, END
    except ImportError: return None
    g=StateGraph(State); g.add_node('router',router_node); g.add_node('retrieval',retrieval_node); g.add_node('prediction',prediction_node)
    g.add_node('respond',lambda s:{**s,'final_response':gemini_phrase(s)}); g.add_node('refusal',lambda s:{**s,'final_response':format_deterministic(s)})
    g.set_entry_point('router'); g.add_conditional_edges('router',lambda s:s['intent'],{'retrieval':'retrieval','prediction':'prediction','off_topic':'refusal'})
    g.add_edge('retrieval','respond'); g.add_edge('prediction','respond'); g.add_edge('respond',END); g.add_edge('refusal',END); return g.compile()

APP=build_graph()
def ask(message,conversation_id='default'):
    start=time.perf_counter();
    if not rate_limiter.allow(conversation_id): return {'response':'Rate limit reached. Please wait before sending more requests.','prediction_metadata':None,'latency_ms':round((time.perf_counter()-start)*1000,1),'trace':[]}
    mem=MEMORY.setdefault(conversation_id,{}); hist=HISTORY.setdefault(conversation_id,[])
    st={'user_query':message,'conversation_id':conversation_id,'history':hist[-12:],'memory':mem,'trace':[]}
    if APP: out=APP.invoke(st)
    else:
        out=router_node(st); out={**out,'final_response':format_deterministic(out)} if out['intent']=='off_topic' else (prediction_node(out) if out['intent']=='prediction' else retrieval_node(out));
        if out['intent']!='off_topic': out['final_response']=gemini_phrase(out)
    hist.extend([('user',message),('assistant',out['final_response'])]); out['latency_ms']=round((time.perf_counter()-start)*1000,1)
    return {'response':out['final_response'],'prediction_metadata':out.get('prediction_metadata'),'latency_ms':out['latency_ms'],'trace':out.get('trace',[])}

if __name__=='__main__':
    for q in ['How is Carlton Blues going lately?','Who will win if Carlton plays Collingwood?','What is the weather today?']:
        x=ask(q,'demo'); print('\nUSER:',q,'\nASSISTANT:',x['response'])
