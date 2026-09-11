from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import ClientRequest, run_agent, approve

app = FastAPI(title="ClientOnboard AI", version="1.0")

class Approval(BaseModel):
    approved: bool

@app.get("/")
def root():
    return {"service":"ClientOnboard AI","status":"running"}

@app.post("/onboard")
def onboard(request: ClientRequest):
    return run_agent(request)

@app.post("/approve/{run_id}")
def approval(run_id: str, body: Approval):
    return approve(run_id, body.approved)
