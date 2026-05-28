from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_logic import answer_agent_question
from business_impact import calculate_business_impact

DATA_DIR = Path(__file__).resolve().parent / "data"

app = FastAPI(
    title="HEIA API",
    description="Hidden Entrepreneur Intelligence Agent — Mastercard Kazakhstan demo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _csv_path(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        raise HTTPException(status_code=500, detail=f"Missing data file: {name}")
    return path


def _read_csv(name: str) -> list[dict[str, Any]]:
    df = pd.read_csv(_csv_path(name))
    return json.loads(df.to_json(orient="records"))


FEEDBACK_STATUSES = (
    "Not contacted",
    "Contacted",
    "Converted",
    "Not interested",
    "False positive",
    "Needs follow-up",
    "Already business customer",
    "Accepted",
    "Rejected",
    "Deferred",
    "Pending",
)

class FeedbackCreate(BaseModel):
    card_id: str
    user_role: str
    feedback_status: str
    comment: str = ""


class AuditLogCreate(BaseModel):
    card_id: str
    user_role: str
    old_status: str
    new_status: str
    timestamp: str | None = None


class AgentQuestion(BaseModel):
    question: str
    role: str = "guest"


class BusinessImpactCalculate(BaseModel):
    high_opportunity_customers: int = Field(..., ge=0)
    conversion_rate_pct: float = Field(..., ge=1, le=30)
    avg_annual_revenue_kzt: float = Field(..., ge=0)
    campaign_cost_kzt: float = Field(..., ge=0)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "HEIA"}


@app.get("/api/scores")
def get_scores(card_id: str | None = None) -> list[dict[str, Any]]:
    rows = _read_csv("demo_cardholder_scores.csv")
    if card_id:
        filtered = [r for r in rows if r.get("card_id") == card_id]
        if not filtered:
            raise HTTPException(status_code=404, detail=f"Card not found: {card_id}")
        return filtered
    return rows


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    path = DATA_DIR / "demo_model_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=500, detail="Missing demo_model_metrics.json")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/features")
def get_features() -> list[dict[str, Any]]:
    return _read_csv("demo_feature_importance.csv")


@app.get("/api/segments")
def get_segments() -> list[dict[str, Any]]:
    return _read_csv("demo_segment_summary.csv")


@app.get("/api/feedback")
def get_feedback() -> list[dict[str, Any]]:
    return _read_csv("demo_feedback_log.csv")


@app.post("/api/audit-log")
def post_audit_log(body: AuditLogCreate) -> dict[str, Any]:
    path = DATA_DIR / "demo_audit_log.csv"
    if not path.exists():
        pd.DataFrame(
            columns=["card_id", "user_role", "old_status", "new_status", "timestamp"]
        ).to_csv(path, index=False)
    df = pd.read_csv(path)
    row = {
        "card_id": body.card_id,
        "user_role": body.user_role,
        "old_status": body.old_status,
        "new_status": body.new_status,
        "timestamp": body.timestamp
        or datetime.utcnow().isoformat() + "Z",
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return {"success": True, "entry": row}


@app.post("/api/feedback")
def post_feedback(body: FeedbackCreate) -> dict[str, Any]:
    if body.feedback_status not in FEEDBACK_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feedback_status. Allowed: {', '.join(FEEDBACK_STATUSES)}",
        )
    path = _csv_path("demo_feedback_log.csv")
    df = pd.read_csv(path)
    row = {
        "card_id": body.card_id,
        "user_role": body.user_role,
        "feedback_status": body.feedback_status,
        "comment": body.comment,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

    scores_path = DATA_DIR / "demo_cardholder_scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        mask = scores["card_id"] == body.card_id
        if mask.any():
            scores.loc[mask, "feedback_status"] = body.feedback_status
            scores.to_csv(scores_path, index=False)

    return {"success": True, "feedback": row}


@app.post("/api/business-impact/calculate")
def post_business_impact_calculate(
    body: BusinessImpactCalculate,
) -> dict[str, Any]:
    return calculate_business_impact(
        high_opportunity_customers=body.high_opportunity_customers,
        conversion_rate_pct=body.conversion_rate_pct,
        avg_annual_revenue_kzt=body.avg_annual_revenue_kzt,
        campaign_cost_kzt=body.campaign_cost_kzt,
    )


def _log_agent_question(question: str, role: str) -> None:
    path = DATA_DIR / "questions_log.csv"
    if not path.exists():
        pd.DataFrame(columns=["timestamp", "role", "question"]).to_csv(path, index=False)
    df = pd.read_csv(path)
    row = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "role": role,
        "question": question,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


@app.post("/api/agent/question")
def agent_question(body: AgentQuestion) -> dict[str, Any]:
    result = answer_agent_question(body.question)
    _log_agent_question(body.question, body.role)
    return {
        "question": body.question,
        "answer": result["answer"],
        "supporting_data": result["supporting_data"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
