from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import google.generativeai as genai
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from business_impact import calculate_business_impact, calculate_business_impact_v2
from agent_logic import (
    answer_agent_question,
    build_intent_focus_block,
    detect_intent,
    director_requests_cardholder_detail,
    intent_supporting_data,
)
from data_loader import (
    CARDHOLDER_BLOCKED_ROLES,
    DATA_DIR,
    append_feedback_row,
    audit_log_records_sorted,
    audit_write_path,
    build_request_agent_context,
    build_system_context,
    feedback_log_records_sorted,
    format_agent_system_context,
    print_data_load_summary,
    verify_required_data_files,
    join_cardholder,
    load_feature_df,
    load_model_metrics_raw,
    load_segments_df,
    normalize_bank_records,
    normalize_model_metrics,
    normalize_product_records,
    normalize_segment_records,
    query_leads,
    query_scores,
    read_csv_records,
    read_json_optional,
    risk_segment_for,
    scores_write_path,
    store,
)

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def load_backend_env() -> bool:
    """Load backend/.env for keys not already set in the process environment.

    Platform secrets (e.g. Render dashboard) take precedence over the file so a
    stale committed or baked-in .env cannot override production credentials.
    """
    env_path = BACKEND_DIR / ".env"
    if not env_path.is_file():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or os.environ.get(key, "").strip():
            continue
        os.environ[key] = value
    return True


def _read_env_file_value(name: str) -> str:
    env_path = BACKEND_DIR / ".env"
    if not env_path.is_file():
        return ""
    prefix = f"{name}="
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(prefix):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return ""


def _env_key_fingerprint(key: str) -> str | None:
    key = key.strip()
    if not key:
        return None
    if len(key) <= 8:
        return f"{key[:2]}…{key[-2:]}"
    return f"{key[:4]}…{key[-4:]}"


def gemini_model_name() -> str:
    load_backend_env()
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def gemini_api_key() -> str:
    load_backend_env()
    return os.environ.get("GEMINI_API_KEY", "").strip()


def ensure_gemini_configured() -> bool:
    key = gemini_api_key()
    if not key:
        return False
    genai.configure(api_key=key)
    return True


load_backend_env()

app = FastAPI()
app.title = "HEIS API"
app.description = "Hidden Entrepreneur Intelligence System — Mastercard Kazakhstan demo"
app.version = "1.0.0"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

AGENT_CONTEXT_TEMPLATE: str | None = None

DEFAULT_SCORES_LIMIT = 500


@app.on_event("startup")
def _startup_load_data() -> None:
    global AGENT_CONTEXT_TEMPLATE

    verify_required_data_files()
    store.load_all()
    print_data_load_summary()

    AGENT_CONTEXT_TEMPLATE = build_system_context(
        store.scores_df,
        store.segment_df,
        store.feature_df,
        store.model_metrics or {},
        kpi=store.kpi,
        banks_df=store.banks_df,
        products_df=store.products_df,
        feedback_df=store.feedback_df,
        assumptions=store.assumptions,
        cluster_df=store.cluster_df,
    )

    env_file = BACKEND_DIR / ".env"
    if load_backend_env():
        print(f"Loaded env from {env_file}")
    elif env_file.is_file():
        print(f"Env file present: {env_file}")
    else:
        print(f"No {env_file} — copy backend/.env.example to backend/.env")

    if ensure_gemini_configured():
        print(f"Gemini agent enabled (model={gemini_model_name()})")
    else:
        print(
            "Gemini agent disabled: set GEMINI_API_KEY in backend/.env or environment "
            "(static rule-based answers will be used)"
        )


FEEDBACK_STATUSES = (
    "Not reviewed",
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
    comment: str = ""


class AgentQuestion(BaseModel):
    question: str
    role: str = "guest"


class BusinessImpactCalculate(BaseModel):
    high_opportunity_customers: int = Field(..., ge=0)
    conversion_rate_pct: float = Field(..., ge=1, le=30)
    avg_annual_revenue_kzt: float = Field(..., ge=0)
    campaign_cost_kzt: float = Field(..., ge=0)


class BusinessImpactCalculateV2(BaseModel):
    leads_count: int = Field(..., ge=0)
    conversion_rate: float = Field(..., ge=0, le=1)
    avg_revenue_per_customer: float = Field(..., ge=0)
    campaign_cost: float = Field(..., ge=0)


def _segments_payload() -> list[dict[str, Any]]:
    df = store.segment_df if store.segment_df is not None else load_segments_df()
    return normalize_segment_records(df, store.assumptions)


def _banks_payload() -> list[dict[str, Any]]:
    if store.banks_df is not None:
        records = json.loads(store.banks_df.to_json(orient="records"))
        return normalize_bank_records(records)
    return []


def _products_payload() -> list[dict[str, Any]]:
    if store.products_df is None:
        raise HTTPException(
            status_code=404, detail="product_opportunity_summary.csv not found"
        )
    records = json.loads(store.products_df.to_json(orient="records"))
    return normalize_product_records(records)


def _kpi_payload() -> dict[str, Any]:
    if store.kpi is not None:
        return store.kpi
    segments = _segments_payload()
    scores = store.scores_df
    if scores is None:
        raise HTTPException(status_code=500, detail="Scores data not loaded")
    high = next((s for s in segments if s["risk_segment"] == "High"), None)
    return {
        "total_scored_consumers": len(scores),
        "top_1_percent_candidates": high["number_of_cardholders"] if high else 0,
        "estimated_total_opportunity_kzt": sum(
            s["estimated_opportunity_value"] for s in segments
        ),
        "average_score": float(scores["commercial_activity_score"].mean()),
    }


def _role_blocked_for_cardholder(role: str | None) -> bool:
    if not role:
        return False
    return role.strip().lower() in CARDHOLDER_BLOCKED_ROLES


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "HEIS"}


# --- Official data package endpoints ---


@app.get("/api/kpis")
def get_kpis(role: str | None = Query(None)) -> dict[str, Any]:
    _ = role
    return _kpi_payload()


@app.get("/api/segments")
def get_segments() -> list[dict[str, Any]]:
    return _segments_payload()


@app.get("/api/banks/opportunities")
def get_bank_opportunities() -> list[dict[str, Any]]:
    banks = _banks_payload()
    if not banks:
        raise HTTPException(
            status_code=404, detail="bank_opportunity_summary.csv not found"
        )
    return banks


@app.get("/api/products/recommendations")
def get_product_recommendations() -> list[dict[str, Any]]:
    return _products_payload()


@app.get("/api/model/metrics")
def get_model_metrics() -> dict[str, Any]:
    if store.model_metrics is not None:
        return store.model_metrics
    return normalize_model_metrics(load_model_metrics_raw())


@app.get("/api/model/features")
def get_model_features() -> list[dict[str, Any]]:
    if store.feature_df is not None:
        return json.loads(store.feature_df.to_json(orient="records"))
    preferred, fallback = "feature_importance.csv", "demo_feature_importance.csv"
    return read_csv_records(preferred, fallback)


@app.get("/api/leads")
def get_leads(
    role: str | None = Query(None),
    bank: str | None = Query(None),
    segment: str | None = Query(None),
    feedback: str | None = Query(None),
    search: str | None = Query(None),
    sort: str | None = Query(None),
    limit: int = Query(500, ge=1, le=10000),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    return query_leads(
        role=role,
        bank=bank or None,
        segment=segment or None,
        feedback=feedback or None,
        search=search or None,
        sort=sort or None,
        limit=limit,
        offset=offset,
    )


@app.get("/api/cardholder/{card_id}")
def get_cardholder(
    card_id: str,
    role: str | None = Query(None),
) -> dict[str, Any]:
    if _role_blocked_for_cardholder(role):
        raise HTTPException(
            status_code=403,
            detail="Aggregated view only for this role",
        )

    profile = join_cardholder(card_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Card not found: {card_id}")
    return profile


@app.get("/api/impact/assumptions")
def get_impact_assumptions() -> dict[str, Any]:
    if store.assumptions is None:
        raise HTTPException(
            status_code=404, detail="business_impact_assumptions.json not found"
        )
    return store.assumptions


@app.get("/api/audit-log")
def get_audit_log() -> list[dict[str, Any]]:
    return audit_log_records_sorted()


@app.post("/api/feedback")
def post_feedback(body: FeedbackCreate) -> dict[str, Any]:
    if body.feedback_status not in FEEDBACK_STATUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feedback_status. Allowed: {', '.join(FEEDBACK_STATUSES)}",
        )

    row = {
        "card_id": body.card_id,
        "user_role": body.user_role,
        "feedback_status": body.feedback_status,
        "comment": body.comment,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    append_feedback_row(row)
    store.refresh_feedback()

    scores_path = scores_write_path()
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        mask = scores["card_id"] == body.card_id
        if mask.any():
            scores.loc[mask, "feedback_status"] = body.feedback_status
            scores.to_csv(scores_path, index=False)
            store.refresh_scores()

    detail_path = DATA_DIR / "cardholder_detail.csv"
    if detail_path.exists():
        detail = pd.read_csv(detail_path)
        if "feedback_status" in detail.columns:
            mask = detail["card_id"] == body.card_id
            if mask.any():
                detail.loc[mask, "feedback_status"] = body.feedback_status
                detail.to_csv(detail_path, index=False)
                store.detail_df = pd.read_csv(detail_path)

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


# --- Legacy aliases (frontend compatibility) ---


@app.get("/api/scores")
def get_scores(
    card_id: str | None = None,
    risk_segment: str | None = None,
    limit: int | None = Query(None, ge=1, le=10000),
    offset: int = Query(0, ge=0),
) -> list[dict[str, Any]]:
    effective_limit = limit
    if card_id:
        effective_limit = None
    elif effective_limit is None:
        effective_limit = DEFAULT_SCORES_LIMIT

    rows = query_scores(
        card_id=card_id,
        risk_segment=risk_segment,
        limit=effective_limit,
        offset=offset,
    )
    if card_id and not rows:
        raise HTTPException(status_code=404, detail=f"Card not found: {card_id}")
    return rows


@app.get("/api/cardholder/{card_id}/detail")
def get_cardholder_detail_legacy(
    card_id: str,
    role: str | None = Query(None),
) -> dict[str, Any]:
    return get_cardholder(card_id, role=role)


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    return get_model_metrics()


@app.get("/api/features")
def get_features() -> list[dict[str, Any]]:
    return get_model_features()


@app.get("/api/kpi")
def get_kpi() -> dict[str, Any]:
    return get_kpis()


@app.get("/api/banks")
def get_banks() -> list[dict[str, Any]]:
    banks = _banks_payload()
    if banks:
        return banks

    scores = store.scores_df
    if scores is None:
        raise HTTPException(status_code=500, detail="Scores data not loaded")
    scores = scores.copy()
    scores["risk_segment"] = scores.apply(
        lambda r: risk_segment_for(
            {
                "opportunity_segment": r.get("opportunity_segment"),
                "risk_segment": r.get("risk_segment"),
            }
        ),
        axis=1,
    )
    high = scores[scores["risk_segment"] == "High"]
    grouped = (
        high.groupby("bank_name")
        .agg(
            number_of_cardholders=("card_id", "count"),
            average_score=("commercial_activity_score", "mean"),
            estimated_opportunity_value=("expected_value_kzt", "sum"),
        )
        .reset_index()
    )
    grouped["high_priority_count"] = grouped["number_of_cardholders"]
    records = json.loads(grouped.to_json(orient="records"))
    return normalize_bank_records(records)


@app.get("/api/products")
def get_products() -> list[dict[str, Any]]:
    return get_product_recommendations()


@app.get("/api/business-impact/assumptions")
def get_business_impact_assumptions() -> dict[str, Any]:
    return get_impact_assumptions()


@app.get("/api/agent/demo-answers")
def get_agent_demo_answers() -> dict[str, str]:
    data = read_json_optional("ai_agent_demo_answers.json")
    if data is None:
        raise HTTPException(status_code=404, detail="ai_agent_demo_answers.json not found")
    return data


@app.get("/api/feedback")
def get_feedback() -> list[dict[str, Any]]:
    return feedback_log_records_sorted()


@app.post("/api/audit-log")
def post_audit_log(body: AuditLogCreate) -> dict[str, Any]:
    path = audit_write_path()
    if not path.exists():
        pd.DataFrame(
            columns=[
                "card_id",
                "user_role",
                "old_status",
                "new_status",
                "timestamp",
                "comment",
            ]
        ).to_csv(path, index=False)
    df = pd.read_csv(path)
    row = {
        "card_id": body.card_id,
        "user_role": body.user_role,
        "old_status": body.old_status,
        "new_status": body.new_status,
        "timestamp": body.timestamp or datetime.utcnow().isoformat() + "Z",
        "comment": body.comment,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)
    return {"success": True, "entry": row}


@app.post("/api/business-impact/calculate/legacy")
def post_business_impact_calculate_legacy(
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


def _demo_answer_for_question(question: str) -> str | None:
    answers = read_json_optional("ai_agent_demo_answers.json")
    if not answers:
        return None
    q = question.lower()
    if "opportunity" in q or "total" in q:
        return answers.get("total_opportunity")
    if "why" in q or "priority" in q or "flagged" in q:
        return answers.get("why_high_priority")
    if "next" in q or "step" in q or "recommend" in q:
        return answers.get("recommended_next_step")
    if "limit" in q or "model" in q:
        return answers.get("model_limitation")
    return None


def _agent_response(
    question: str,
    answer: str,
    supporting_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "supporting_data": supporting_data or {},
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def _gemini_finish_reason(response: Any) -> str | None:
    for candidate in getattr(response, "candidates", None) or []:
        reason = getattr(candidate, "finish_reason", None)
        if reason is None:
            continue
        name = getattr(reason, "name", None) or str(reason)
        if name not in ("STOP", "FINISH_REASON_STOP", "1"):
            return str(name)
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = getattr(prompt_feedback, "block_reason", None)
    if block_reason:
        return f"blocked:{block_reason}"
    return None


def _extract_gemini_text(response: Any) -> str:
    try:
        text = response.text
        if text and str(text).strip():
            return str(text).strip()
    except (ValueError, AttributeError):
        pass

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text and str(part_text).strip():
                return str(part_text).strip()
    return ""


def _call_gemini(system_context: str, user_message: str) -> tuple[str, str | None]:
    primary = gemini_model_name()
    fallbacks = [primary]
    if primary != "gemini-flash-latest":
        fallbacks.append("gemini-flash-latest")

    last_error: str | None = None
    for model_name in fallbacks:
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=system_context,
            )
            response = model.generate_content(
                user_message,
                request_options={"timeout": 90},
            )
            answer = _extract_gemini_text(response)
            if answer:
                return answer, None
            finish = _gemini_finish_reason(response)
            last_error = finish or "Gemini returned an empty response"
        except Exception as exc:
            last_error = str(exc)
            print(f"[HEIS] Gemini error ({model_name}): {last_error}")

    return "", last_error or "Gemini request failed"


@app.get("/api/agent/status")
def agent_status() -> dict[str, Any]:
    key = gemini_api_key()
    file_key = _read_env_file_value("GEMINI_API_KEY")
    env_path = BACKEND_DIR / ".env"
    if key and file_key and key == file_key:
        key_source = "env_file"
    elif key:
        key_source = "environment"
    else:
        key_source = "none"
    return {
        "gemini_configured": bool(key),
        "model": gemini_model_name(),
        "context_ready": AGENT_CONTEXT_TEMPLATE is not None,
        "env_file": str(env_path),
        "env_file_exists": env_path.is_file(),
        "gemini_key_fingerprint": _env_key_fingerprint(key),
        "env_file_gemini_key_fingerprint": _env_key_fingerprint(file_key),
        "gemini_key_source": key_source,
    }


@app.post("/api/agent/question")
def agent_question(body: AgentQuestion) -> dict[str, Any]:
    if not AGENT_CONTEXT_TEMPLATE:
        raise HTTPException(status_code=500, detail="Agent context not initialized")

    _log_agent_question(body.question, body.role)
    role = (body.role or "guest").strip()

    if director_requests_cardholder_detail(role, body.question):
        return _agent_response(
            body.question,
            "Your role allows aggregated insights only.",
            {"blocked": True, "reason": "director_cardholder_detail"},
        )

    intent = detect_intent(body.question)
    supporting = intent_supporting_data(intent, body.question) if intent else {}
    intent_block = build_intent_focus_block(intent, body.question) if intent else ""

    if ensure_gemini_configured():
        model_name = gemini_model_name()
        system_context = format_agent_system_context(AGENT_CONTEXT_TEMPLATE, role)
        request_context = build_request_agent_context(
            question=body.question,
            role=role,
            intent_block=intent_block,
        )
        user_message_parts = []
        if request_context.strip():
            user_message_parts.append(request_context)
        user_message_parts.append(f"User question: {body.question}")
        user_message = "\n\n".join(user_message_parts)

        try:
            answer, gemini_error = _call_gemini(system_context, user_message)
        except Exception as exc:
            answer = ""
            gemini_error = str(exc)

        if gemini_error or not answer.strip():
            result = answer_agent_question(body.question, role=role, intent=intent)
            answer = result["answer"]
            supporting = {
                **supporting,
                **result.get("supporting_data", {}),
                "source": "rules",
                "gemini_model": model_name,
            }
            if gemini_error:
                supporting["gemini_error"] = gemini_error
        else:
            supporting = {
                **supporting,
                "source": "gemini",
                "gemini_model": model_name,
            }

        if intent:
            supporting.setdefault("intent", intent)
        return _agent_response(body.question, answer, supporting)

    demo_answer = _demo_answer_for_question(body.question)
    if demo_answer:
        return _agent_response(
            body.question,
            demo_answer,
            {"source": "demo", **supporting},
        )

    result = answer_agent_question(body.question, role=role, intent=intent)
    merged_supporting = {**supporting, **result.get("supporting_data", {}), "source": "rules"}
    if intent:
        merged_supporting.setdefault("intent", intent)
    return _agent_response(body.question, result["answer"], merged_supporting)
