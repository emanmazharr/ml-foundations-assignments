"""FastAPI wrapper for the Day 5 AFL assistant."""
import logging, json, time, math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from afl_langgraph_app import ask
logging.basicConfig(filename='afl_monitor.log', level=logging.INFO, format='%(asctime)s %(message)s')
app=FastAPI(title='AFL Assistant API', version='1.0')
class ChatRequest(BaseModel):
    message:str=Field(min_length=1,max_length=2000)
    conversation_id:str=Field(default='default',min_length=1,max_length=100)

def _sanitize(obj):
    """Recursively replace NaN/Infinity with None -- raw JSON does not support
    them, and pandas/numpy stats can legitimately be NaN for missing data.
    Fixes a bug found during pre-submission testing: any response containing
    a NaN stat (e.g. a player missing one field) crashed the endpoint with
    'ValueError: Out of range float values are not JSON compliant'."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj

@app.get('/health')
def health(): return {'status':'ok','service':'AFL Assistant'}
@app.post('/chat')
def chat(req:ChatRequest):
    t=time.perf_counter()
    try:
        result=ask(req.message,req.conversation_id)
        result=_sanitize(result)
        logging.info(json.dumps({'query':req.message,'conversation_id':req.conversation_id,'latency_ms':result['latency_ms'],'tools':[x.get('tool') for x in result.get('trace',[]) if x.get('tool')],'prediction':bool(result.get('prediction_metadata'))}))
        return result
    except Exception as e:
        logging.exception('request failed')
        raise HTTPException(status_code=500,detail='The AFL assistant encountered an internal error; please retry.')

