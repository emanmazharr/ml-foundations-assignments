import os, re, json
from pathlib import Path
from typing import List
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from afl_data_tools import get_player_season_stats, get_player_recent_rounds, compare_player_recent_to_career, get_team_record_vs_opponent, player_search

SYSTEM_PROMPT = """You are an AFL-only conversational assistant. Your scope is Australian Football League (AFL) teams, players, matches, statistics, history, rules, competitions and dataset-grounded analysis. Do not answer questions about other sports, general trivia, politics, coding, entertainment, personal advice, or unrelated chit-chat. For an out-of-scope request, politely refuse in one sentence and redirect to AFL.

GROUNDING RULES: For factual statistics, records, player totals, averages, results, or match details, you MUST call the relevant retrieval tool. Never invent or rely on remembered numbers. Only state numerical facts that appear in the returned tool output. If the dataset does not contain the requested value, say so and do not guess. When context is needed for a follow-up, use the conversation history, but still retrieve the new factual answer from a tool.

Useful follow-ups may omit the player/team name; infer it only from the conversation history. Be concise and transparent about what the dataset supports."""

@tool
def afl_player_season_stats(player: str, season: int) -> dict:
    """Get exact AFL player season totals and averages from the supplied seasonal statistics dataset."""
    return get_player_season_stats(player, season)

@tool
def afl_player_recent_rounds(player: str, n: int=1, before_season: int=None, before_round: int=None) -> dict:
    """Get a player's most recent AFL match-level rows, optionally the games immediately before a specified round."""
    return get_player_recent_rounds(player,n,before_season,before_round)

@tool
def afl_player_career_comparison(player: str, stat: str='disposals', recent_n: int=5) -> dict:
    """Compare a player's recent average for a statistic with their career average computed from the real match-level dataset."""
    return compare_player_recent_to_career(player,stat,recent_n)

@tool
def afl_team_head_to_head(team: str, opponent: str) -> dict:
    """Get exact historical AFL head-to-head meetings, wins, losses and draws between two teams."""
    return get_team_record_vs_opponent(team,opponent)

@tool
def afl_player_search(name: str) -> dict:
    """Find matching AFL players in the supplied player-information dataset."""
    return {'matches': player_search(name)}

TOOLS=[afl_player_season_stats,afl_player_recent_rounds,afl_player_career_comparison,afl_team_head_to_head,afl_player_search]

def scope_check(text):
    t=text.lower().strip()
    off=['cricket','soccer','football nfl','nba','tennis','recipe','bitcoin','politics','python','javascript','movie','weather','joke','homework','relationship','stock market']
    afl=['afl','australian football','aussie rules','football','player','team','round','match','disposal','goal','mark','tackle','premiership','grand final','brownlow']
    if any(x in t for x in off) and not ('afl' in t or 'australian football' in t or 'aussie rules' in t): return False
    return bool(any(x in t for x in afl))

class AFLChatAgent:
    def __init__(self, model=None):
        key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not key: raise RuntimeError('Set GEMINI_API_KEY (or GOOGLE_API_KEY) before starting the LangChain agent.')
        llm=ChatGoogleGenerativeAI(model=model or os.getenv('GEMINI_MODEL','gemini-3.6-flash'), temperature=0, google_api_key=key)
        prompt=SystemMessage(content=SYSTEM_PROMPT)
        # Agent prompt must expose a system message plus chat history placeholder.
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        p=ChatPromptTemplate.from_messages([('system',SYSTEM_PROMPT),MessagesPlaceholder('chat_history'),('human','{input}'),MessagesPlaceholder('agent_scratchpad')])
        agent=create_tool_calling_agent(llm,TOOLS,p)
        self.executor=AgentExecutor(agent=agent,tools=TOOLS,verbose=False,return_intermediate_steps=True)
        self.history=[]
        self.last_tool_results=[]
    def ask(self,user_text):
        if not scope_check(user_text):
            return 'I can help with AFL teams, players, matches, stats, history and rules. Please ask me an AFL-related question.'
        result=self.executor.invoke({'input':user_text,'chat_history':self.history})
        self.last_tool_results=[]
        for action, observation in result.get('intermediate_steps',[]):
            self.last_tool_results.append(observation)
        answer=result['output']
        self.history += [HumanMessage(content=user_text),AIMessage(content=answer)]
        return answer

def demo():
    a=AFLChatAgent();
    for q in ['Who is in the player dataset named Gary Ablett?','What were Gary Ablett\'s season stats in 2019?','What about the round before that?','How does that compare with his career average?']:
        print('USER:',q); print('ASSISTANT:',a.ask(q)); print()

if __name__=='__main__': demo()
