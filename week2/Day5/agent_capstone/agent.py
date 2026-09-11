import json, logging, time, uuid
from pathlib import Path
from pydantic import BaseModel, Field, EmailStr, ValidationError

logging.basicConfig(filename="agent.log", level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")

DATA_FILE = Path(__file__).parent / "data" / "services.json"

class ClientRequest(BaseModel):
    client_name: str = Field(min_length=2, max_length=80)
    email: EmailStr
    project_description: str = Field(min_length=10, max_length=2000)
    budget: float = Field(gt=0, le=100000)
    deadline_days: int = Field(gt=0, le=365)

def load_services():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logging.exception("service lookup failure")
        raise RuntimeError("Service data source is temporarily unavailable") from e

def service_lookup(text: str):
    services = load_services()
    t = text.lower()
    scored = []
    for s in services:
        score = sum(1 for k in s["keywords"] if k in t)
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored and scored[0][0] > 0 else services[0]

def safety_check(text: str):
    risky = ["steal password", "hack account", "malware", "phishing", "credit card theft"]
    if any(x in text.lower() for x in risky):
        return False, "I cannot assist with harmful, illegal, or credential-theft activities."
    return True, ""

def analyze(req, service):
    desc = req.project_description.lower()
    priority = "High" if req.deadline_days <= 7 else "Normal"
    fit = "Good" if req.budget >= service["price_from"] else "Budget may be below the recommended starting price."
    return {
        "project_type": service["name"],
        "priority": priority,
        "budget_fit": fit,
        "estimated_starting_price": service["price_from"],
        "estimated_delivery_days": service["delivery_days"],
        "notes": "Clarify scope, required pages/features, content, hosting, and revision limits before final quotation."
    }

def run_agent(req: ClientRequest):
    run_id = str(uuid.uuid4())
    start = time.perf_counter()
    logging.info("run=%s input client=%s budget=%s deadline=%s", run_id, req.client_name, req.budget, req.deadline_days)

    ok, reason = safety_check(req.project_description)
    if not ok:
        logging.warning("run=%s safety refusal", run_id)
        return {"run_id": run_id, "status": "refused", "message": reason, "human_approval_required": False}

    try:
        service = service_lookup(req.project_description)
    except RuntimeError as e:
        return {"run_id": run_id, "status": "error", "message": str(e), "human_approval_required": False}

    analysis = analyze(req, service)
    proposal = {
        "client": req.client_name,
        "email": str(req.email),
        "recommended_service": service["name"],
        "summary": req.project_description,
        "budget": req.budget,
        "deadline_days": req.deadline_days,
        "analysis": analysis,
        "draft_message": f"Hello {req.client_name}, based on your requirements, we recommend {service['name']}. "
                         f"The starting estimate is ${service['price_from']} with an estimated delivery of "
                         f"{service['delivery_days']} days. Final pricing will be confirmed after scope review."
    }
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    logging.info("run=%s tool=service_lookup latency_ms=%s status=pending_approval", run_id, latency_ms)
    return {"run_id": run_id, "status": "pending_approval",
            "human_approval_required": True, "latency_ms": latency_ms,
            "proposal": proposal}

APPROVALS = {}
def approve(run_id: str, approved: bool):
    APPROVALS[run_id] = approved
    return {"run_id": run_id, "approved": approved,
            "status": "approved" if approved else "rejected"}
