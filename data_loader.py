from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

DATA_DIR = Path(__file__).resolve().parent / "data"

RISK_TIERS = ("High", "Medium", "Low")

OPPORTUNITY_TO_RISK: dict[str, str] = {
    "Top 1% highest priority": "High",
    "Top 5% priority": "High",
    "Top 10% review": "Medium",
    "Standard monitoring": "Low",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
}

DEFAULT_RECOMMENDED_PRODUCT: dict[str, str] = {
    "High": "Offer SME digital business card package",
    "Medium": "Relationship manager review",
    "Low": "Monitor only",
}

BAND_RECOMMENDED_PRODUCT: dict[str, str] = {
    "Top 1% highest priority": "Offer SME digital business card package",
    "Top 5% priority": "Offer e-commerce acquiring / business card",
    "Top 10% review": "Relationship manager review",
    "Standard monitoring": "Monitor only",
}

OPPORTUNITY_BAND_ORDER: tuple[str, ...] = (
    "Top 1% highest priority",
    "Top 5% priority",
    "Top 10% review",
    "Standard monitoring",
)

REQUIRED_DATA_FILES = (
    "cardholder_scores.csv",
    "cardholder_detail.csv",
    "model_metrics.json",
    "feature_importance.csv",
    "segment_summary.csv",
    "bank_opportunity_summary.csv",
    "product_opportunity_summary.csv",
    "kpi_summary.json",
    "business_impact_assumptions.json",
    "feedback_log.csv",
)

FEEDBACK_DEFAULT_COLUMNS = [
    "card_id",
    "user_role",
    "feedback_status",
    "comment",
    "date",
]

DATA_FILES = {
    "scores": ("cardholder_scores.csv", "demo_cardholder_scores.csv"),
    "segments": ("segment_summary.csv", "demo_segment_summary.csv"),
    "features": ("feature_importance.csv", "demo_feature_importance.csv"),
    "metrics": ("model_metrics.json", "demo_model_metrics.json"),
    "feedback": ("feedback_log.csv", "demo_feedback_log.csv"),
    "audit": ("audit_log.csv", "demo_audit_log.csv"),
    "kpi": ("kpi_summary.json", None),
    "banks": ("bank_opportunity_summary.csv", None),
    "products": ("product_opportunity_summary.csv", None),
    "detail": ("cardholder_detail.csv", None),
    "assumptions": ("business_impact_assumptions.json", None),
    "agent_answers": ("ai_agent_demo_answers.json", None),
    "cluster": ("cluster_summary.csv", None),
}

SENSITIVE_COLUMNS = ("card_number",)

AGGREGATED_LEAD_ROLES = frozenset({"director", "product_manager"})
FULL_LEAD_ROLES = frozenset(
    {
        "sales",
        "sales_manager",
        "risk",
        "risk_compliance",
        "admin",
        "data_scientist",
        "judge_demo",
        "judge",
    }
)

CARDHOLDER_BLOCKED_ROLES = frozenset({"director"})


def resolve_data_path(preferred: str, fallback: str | None = None) -> Path:
    preferred_path = DATA_DIR / preferred
    if preferred_path.exists():
        return preferred_path
    if fallback:
        fallback_path = DATA_DIR / fallback
        if fallback_path.exists():
            return fallback_path
    raise HTTPException(
        status_code=500,
        detail=f"Missing data file: {preferred}"
        + (f" (also tried {fallback})" if fallback else ""),
    )


def read_csv_df(preferred: str, fallback: str | None = None) -> pd.DataFrame:
    return pd.read_csv(resolve_data_path(preferred, fallback))


def read_csv_records(preferred: str, fallback: str | None = None) -> list[dict[str, Any]]:
    df = read_csv_df(preferred, fallback)
    return json.loads(df.to_json(orient="records"))


def read_json_obj(preferred: str, fallback: str | None = None) -> dict[str, Any]:
    path = resolve_data_path(preferred, fallback) if fallback else DATA_DIR / preferred
    if fallback is None and not path.exists():
        raise HTTPException(status_code=500, detail=f"Missing data file: {preferred}")
    if fallback is not None:
        path = resolve_data_path(preferred, fallback)
    return json.loads(path.read_text(encoding="utf-8"))


def read_json_optional(preferred: str) -> dict[str, Any] | None:
    path = DATA_DIR / preferred
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def risk_segment_for(row: dict[str, Any] | pd.Series) -> str:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    if row.get("risk_segment") in RISK_TIERS:
        return str(row["risk_segment"])
    opp = row.get("opportunity_segment") or row.get("risk_segment") or ""
    return OPPORTUNITY_TO_RISK.get(str(opp), "Low")


def strip_sensitive_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in SENSITIVE_COLUMNS}


def normalize_cardholder_record(row: dict[str, Any]) -> dict[str, Any]:
    out = strip_sensitive_columns(dict(row))
    opp = out.get("opportunity_segment") or out.get("risk_segment") or ""
    tier = risk_segment_for(out)
    out["opportunity_segment"] = str(opp) if opp else tier
    out["risk_segment"] = tier
    if pd.isna(out.get("feedback_status")):
        out["feedback_status"] = "Not reviewed"
    return out


def _conversion_rate_for_opportunity(
    opportunity_segment: str, assumptions: dict[str, Any] | None
) -> float:
    rates = (assumptions or {}).get("conversion_rates", {})
    if opportunity_segment in rates:
        return float(rates[opportunity_segment])
    tier = OPPORTUNITY_TO_RISK.get(opportunity_segment, "Low")
    demo_defaults = {"High": 0.34, "Medium": 0.21, "Low": 0.09}
    return float(demo_defaults.get(tier, 0.1))


def normalize_segment_records(
    df: pd.DataFrame, assumptions: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    if "risk_segment" in df.columns and "recommended_product" in df.columns:
        records: list[dict[str, Any]] = []
        for row in json.loads(df.to_json(orient="records")):
            rec = dict(row)
            if "opportunity_segment" not in rec:
                rec["opportunity_segment"] = rec["risk_segment"]
            records.append(rec)
        return records

    if "opportunity_segment" not in df.columns:
        raise HTTPException(status_code=500, detail="Invalid segment_summary schema")

    records = []
    for _, row in df.iterrows():
        opp_seg = str(row["opportunity_segment"])
        tier = OPPORTUNITY_TO_RISK.get(opp_seg, "Low")
        records.append(
            {
                "risk_segment": tier,
                "opportunity_segment": opp_seg,
                "number_of_cardholders": int(row["number_of_cardholders"]),
                "average_score": float(row["average_score"]),
                "recommended_product": BAND_RECOMMENDED_PRODUCT.get(
                    opp_seg, DEFAULT_RECOMMENDED_PRODUCT[tier]
                ),
                "estimated_conversion_rate": _conversion_rate_for_opportunity(
                    opp_seg, assumptions
                ),
                "estimated_opportunity_value": float(
                    row["estimated_opportunity_value"]
                ),
            }
        )

    order = {band: index for index, band in enumerate(OPPORTUNITY_BAND_ORDER)}
    records.sort(
        key=lambda rec: order.get(str(rec["opportunity_segment"]), len(order))
    )
    return records


def normalize_model_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    if "accuracy" in raw and "roc_auc" in raw:
        out = dict(raw)
        out.setdefault("trained_at", raw.get("training_date") or raw.get("created_at"))
        out.setdefault("training_date", raw.get("trained_at") or raw.get("created_at"))
        if "confusion_matrix" in raw and isinstance(raw["confusion_matrix"], dict):
            cm = raw["confusion_matrix"]
            out["confusion_matrix"] = {
                "tp": int(cm.get("tp", 0)),
                "fp": int(cm.get("fp", 0)),
                "fn": int(cm.get("fn", 0)),
                "tn": int(cm.get("tn", 0)),
            }
        return out

    proxy_list = raw.get("proxy_validation") or []
    proxy = proxy_list[0] if proxy_list else {}
    roc = float(proxy.get("proxy_roc_auc", 0.0))
    pr = float(proxy.get("proxy_pr_auc", roc))
    scored = int(raw.get("number_of_scored_consumer_cards", 0))
    top1 = int(raw.get("top_1_percent_candidates", 0))

    return {
        **raw,
        "model_name": raw.get("model_name", "HEIS Scoring Model"),
        "model_version": raw.get("model_version", "v1"),
        "trained_at": raw.get("created_at") or raw.get("trained_at"),
        "accuracy": roc,
        "precision": pr,
        "recall": roc,
        "f1": (2 * pr * roc / (pr + roc)) if (pr + roc) else 0.0,
        "roc_auc": roc,
        "training_samples": scored,
        "validation_samples": 0,
        "positive_class_rate": (top1 / scored) if scored else 0.0,
        "notes": raw.get("important_note") or raw.get("notes"),
    }


def normalize_bank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in records:
        rec = dict(row)
        if "high_opportunity_count" not in rec and "high_priority_count" in rec:
            rec["high_opportunity_count"] = rec["high_priority_count"]
        out.append(rec)
    return out


def normalize_product_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in records:
        rec = dict(row)
        if "recommended_product" not in rec and "recommended_action" in rec:
            rec["recommended_product"] = rec["recommended_action"]
        out.append(rec)
    return out


@lru_cache(maxsize=1)
def load_scores_df() -> pd.DataFrame:
    if store.scores_df is not None:
        return store.scores_df
    preferred, fallback = DATA_FILES["scores"]
    return read_csv_df(preferred, fallback)


def load_segments_df() -> pd.DataFrame:
    preferred, fallback = DATA_FILES["segments"]
    return read_csv_df(preferred, fallback)


def load_feature_df() -> pd.DataFrame:
    preferred, fallback = DATA_FILES["features"]
    return read_csv_df(preferred, fallback)


def load_model_metrics_raw() -> dict[str, Any]:
    preferred, fallback = DATA_FILES["metrics"]
    return read_json_obj(preferred, fallback)


def scores_write_path() -> Path:
    preferred, fallback = DATA_FILES["scores"]
    preferred_path = DATA_DIR / preferred
    if preferred_path.exists():
        return preferred_path
    return DATA_DIR / fallback


def feedback_write_path() -> Path:
    preferred, fallback = DATA_FILES["feedback"]
    preferred_path = DATA_DIR / preferred
    if preferred_path.exists():
        return preferred_path
    return resolve_data_path(preferred, fallback)


def verify_required_data_files() -> None:
    missing = [name for name in REQUIRED_DATA_FILES if not (DATA_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required data files in {DATA_DIR}: {', '.join(missing)}"
        )


def print_data_load_summary() -> None:
    print("[HEIS] Data load summary:")
    for name in sorted(REQUIRED_DATA_FILES):
        path = DATA_DIR / name
        if not path.exists():
            print(f"  {name}: MISSING")
            continue
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, low_memory=False)
            print(f"  {name}: {len(df)} rows")
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                print(f"  {name}: keys={list(data.keys())}")
            elif isinstance(data, list):
                print(f"  {name}: list with {len(data)} items")
            else:
                print(f"  {name}: {type(data).__name__}")


def append_feedback_row(row: dict[str, Any]) -> None:
    """Append feedback; align columns with existing file instead of failing."""
    path = feedback_write_path()
    if not path.exists():
        df = pd.DataFrame(columns=FEEDBACK_DEFAULT_COLUMNS)
    else:
        df = pd.read_csv(path)

    new_row = pd.DataFrame([row])
    all_columns = list(dict.fromkeys(list(df.columns) + list(new_row.columns)))
    df = df.reindex(columns=all_columns)
    new_row = new_row.reindex(columns=all_columns)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)


def audit_write_path() -> Path:
    preferred, fallback = DATA_FILES["audit"]
    preferred_path = DATA_DIR / preferred
    if preferred_path.exists():
        return preferred_path
    return DATA_DIR / fallback


def _ensure_risk_tiers(df: pd.DataFrame) -> pd.DataFrame:
    if "risk_segment" in df.columns and df["risk_segment"].isin(RISK_TIERS).all():
        return df
    df = df.copy()
    df["risk_segment"] = df.apply(
        lambda r: risk_segment_for(
            {
                "opportunity_segment": r.get("opportunity_segment"),
                "risk_segment": r.get("risk_segment"),
            }
        ),
        axis=1,
    )
    return df


def query_scores(
    *,
    card_id: str | None = None,
    risk_segment: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    df = _ensure_risk_tiers(_scores_dataframe())

    if card_id:
        filtered = df[df["card_id"] == card_id]
        if filtered.empty:
            return []
        return [normalize_cardholder_record(r) for r in json.loads(filtered.to_json(orient="records"))]

    if risk_segment:
        df = df[df["risk_segment"] == risk_segment]

    df = df.sort_values("commercial_activity_score", ascending=False)
    if offset:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.head(limit)

    return [normalize_cardholder_record(r) for r in json.loads(df.to_json(orient="records"))]


def load_cardholder_detail(card_id: str) -> dict[str, Any] | None:
    preferred, _ = DATA_FILES["detail"]
    path = DATA_DIR / preferred
    if not path.exists():
        return None
    df = pd.read_csv(path)
    match = df[df["card_id"] == card_id]
    if match.empty:
        return None
    row = json.loads(match.to_json(orient="records"))[0]
    return normalize_cardholder_record(row)


def format_kpi_summary_text(kpi: dict[str, Any] | None) -> str:
    if not kpi:
        return "N/A"
    lines: list[str] = []
    labels = {
        "total_scored_consumers": "Total scored consumers",
        "top_1_percent_candidates": "Top 1% candidates",
        "top_5_percent_candidates": "Top 5% candidates",
        "top_10_percent_candidates": "Top 10% candidates",
        "estimated_total_opportunity_kzt": "Estimated total opportunity (KZT)",
        "average_score": "Average commercial activity score",
    }
    for key, label in labels.items():
        if key not in kpi:
            continue
        val = kpi[key]
        if key.endswith("_kzt") and isinstance(val, (int, float)):
            lines.append(f"- {label}: ₸{float(val):,.2f}")
        elif isinstance(val, float):
            lines.append(f"- {label}: {val:,.4f}")
        else:
            lines.append(f"- {label}: {val:,}" if isinstance(val, int) else f"- {label}: {val}")
    return "\n".join(lines) if lines else json.dumps(kpi, indent=2)


def feedback_status_summary(feedback_df: pd.DataFrame | None) -> str:
    if feedback_df is None or feedback_df.empty:
        return "No feedback records yet."
    status_col = "feedback_status"
    if status_col not in feedback_df.columns:
        return "No feedback status column in log."
    counts = feedback_df[status_col].fillna("Unknown").astype(str).value_counts()
    return counts.to_string()


_CARD_ID_PATTERN = re.compile(r"\bCARD_\d{4,8}\b", re.IGNORECASE)

AGENT_DATA_FILE_MANIFEST = (
    "cardholder_scores.csv",
    "cardholder_detail.csv",
    "segment_summary.csv",
    "bank_opportunity_summary.csv",
    "product_opportunity_summary.csv",
    "feature_importance.csv",
    "model_metrics.json",
    "kpi_summary.json",
    "business_impact_assumptions.json",
    "feedback_log.csv",
    "technical_submission_scores.csv",
    "cluster_summary.csv",
)


def extract_card_id_from_question(question: str) -> str | None:
    match = _CARD_ID_PATTERN.search(question)
    return match.group(0).upper() if match else None


def _dataframe_block(df: pd.DataFrame | None, *, max_rows: int | None = None) -> str:
    if df is None or df.empty:
        return "N/A"
    view = df if max_rows is None else df.head(max_rows)
    safe = view.copy()
    for col in SENSITIVE_COLUMNS:
        if col in safe.columns:
            safe = safe.drop(columns=[col])
    return safe.to_string()


def build_scores_aggregate_block(scores_df: pd.DataFrame) -> str:
    if scores_df is None or scores_df.empty:
        return "N/A"
    df = scores_df.copy()
    if "risk_segment" not in df.columns or df["risk_segment"].isin(RISK_TIERS).sum() == 0:
        df["risk_segment"] = df.apply(
            lambda r: risk_segment_for(
                {
                    "opportunity_segment": r.get("opportunity_segment"),
                    "risk_segment": r.get("risk_segment"),
                }
            ),
            axis=1,
        )

    lines = [
        f"Total scored cardholders: {len(df):,}",
        f"Average commercial_activity_score: {df['commercial_activity_score'].mean():.6f}",
        f"Score min / max: {df['commercial_activity_score'].min():.6f} / {df['commercial_activity_score'].max():.6f}",
    ]
    if "risk_segment" in df.columns:
        lines.append("\nCounts by risk_segment:")
        lines.append(df["risk_segment"].value_counts().to_string())
    if "opportunity_segment" in df.columns:
        lines.append("\nCounts by opportunity_segment:")
        lines.append(df["opportunity_segment"].value_counts().to_string())
    if "bank_name" in df.columns:
        lines.append("\nTop 10 banks by cardholder count:")
        lines.append(df["bank_name"].value_counts().head(10).to_string())
    if "recommended_action" in df.columns:
        lines.append("\nTop recommended actions:")
        lines.append(df["recommended_action"].value_counts().head(8).to_string())

    sample_cols = [
        c
        for c in (
            "card_id",
            "bank_name",
            "commercial_activity_score",
            "risk_segment",
            "opportunity_segment",
            "recommended_action",
            "expected_value_kzt",
            "feedback_status",
        )
        if c in df.columns
    ]
    if sample_cols:
        top = df.nlargest(20, "commercial_activity_score")[sample_cols]
        lines.append("\nTop 20 cardholders by score (use card_id for follow-up):")
        lines.append(_dataframe_block(top))

    return "\n".join(lines)


def build_technical_submission_block() -> str:
    path = DATA_DIR / "technical_submission_scores.csv"
    if not path.exists():
        return "N/A"
    df = pd.read_csv(path)
    if df.empty:
        return "N/A"
    score_col = "score" if "score" in df.columns else df.columns[-1]
    lines = [
        f"technical_submission_scores.csv rows: {len(df):,}",
        df[score_col].describe().to_string(),
    ]
    if "score" in df.columns or score_col in df.columns:
        top = df.nlargest(10, score_col)
        lines.append(f"\nTop 10 submission scores:\n{_dataframe_block(top)}")
    return "\n".join(lines)


def build_request_agent_context(
    *,
    question: str,
    role: str,
    intent_block: str = "",
) -> str:
    """Per-request context appended to the user message for Gemini."""
    parts: list[str] = []

    if intent_block.strip():
        parts.append(f"FOCUSED DATA FOR THIS QUESTION:\n{intent_block.strip()}")

    card_id = extract_card_id_from_question(question)
    if card_id and role.strip().lower() not in CARDHOLDER_BLOCKED_ROLES:
        profile = join_cardholder(card_id)
        if profile:
            parts.append(
                "CARDHOLDER RECORD FOR THIS QUESTION (from cardholder_scores + cardholder_detail):\n"
                + json.dumps(profile, indent=2, ensure_ascii=False, default=str)
            )
        else:
            parts.append(f"Note: card_id {card_id} was not found in loaded portfolio data.")

    return "\n\n".join(parts)


def build_system_context(
    scores_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model_metrics: dict[str, Any],
    kpi: dict[str, Any] | None = None,
    banks_df: pd.DataFrame | None = None,
    products_df: pd.DataFrame | None = None,
    feedback_df: pd.DataFrame | None = None,
    assumptions: dict[str, Any] | None = None,
    cluster_df: pd.DataFrame | None = None,
) -> str:
    """Data-only context template; role is injected per request via format_agent_system_context."""
    segment_block = _dataframe_block(segment_df)
    banks_block = _dataframe_block(banks_df, max_rows=20)
    products_block = _dataframe_block(products_df)
    features_block = _dataframe_block(feature_df)
    kpi_block = format_kpi_summary_text(kpi)
    feedback_summary = feedback_status_summary(feedback_df)
    feedback_rows = _dataframe_block(feedback_df, max_rows=50)
    assumptions_block = (
        json.dumps(assumptions, indent=2, ensure_ascii=False)
        if assumptions
        else "N/A"
    )
    scores_block = build_scores_aggregate_block(scores_df)
    technical_block = build_technical_submission_block()
    cluster_block = _dataframe_block(cluster_df)
    manifest = "\n".join(f"- {name}" for name in AGENT_DATA_FILE_MANIFEST)

    return f"""
You are HEIS — Hidden Entrepreneur Intelligence System (Mastercard Kazakhstan demo).
The user role for this session is: {{user_role}}

You MUST answer ONLY using the official portfolio data below (loaded from backend/data/).
Do not invent metrics, cardholders, or banks. If the answer is not in the data, say so clearly.

Available data files (full portfolio is summarized here; large tables are aggregated):
{manifest}

=== kpi_summary.json ===
{kpi_block}

=== cardholder_scores.csv (aggregates + top-20 sample; full file has {len(scores_df):,} rows) ===
{scores_block}

=== segment_summary.csv ===
{segment_block}

=== bank_opportunity_summary.csv ===
{banks_block}

=== product_opportunity_summary.csv ===
{products_block}

=== feature_importance.csv ===
{features_block}

=== model_metrics.json ===
{json.dumps(model_metrics, indent=2, ensure_ascii=False)}

=== business_impact_assumptions.json ===
{assumptions_block}

=== feedback_log.csv ===
Summary:
{feedback_summary}

All rows:
{feedback_rows}

=== technical_submission_scores.csv ===
{technical_block}

=== cluster_summary.csv ===
{cluster_block}

Rules:
- Answer in the same language as the question (Russian, English, or Kazakh if asked)
- Always cite specific numbers from the blocks above
- For ROI / conversion scenarios, use business_impact_assumptions.json conversion_rates when relevant
- If the user names a card_id (e.g. CARD_000042), use the CARDHOLDER RECORD section in the user message when present
- If user role is Director and they ask for individual cardholder PII or row-level detail without a permitted use case,
  respond exactly: "Your role allows aggregated insights only."
- Keep answers business-focused and concise (3–8 sentences unless a list is requested)
- Never show raw card numbers (card_number); card_id tokens are allowed
- When comparing segments, banks, or products, quote counts and KZT values from the data
""".strip()


def format_agent_system_context(template: str, user_role: str) -> str:
    return template.replace("{user_role}", user_role or "guest")


class DataStore:
    """In-memory official data package loaded once at API startup."""

    def __init__(self) -> None:
        self.scores_df: pd.DataFrame | None = None
        self.detail_df: pd.DataFrame | None = None
        self.segment_df: pd.DataFrame | None = None
        self.feature_df: pd.DataFrame | None = None
        self.banks_df: pd.DataFrame | None = None
        self.products_df: pd.DataFrame | None = None
        self.cluster_df: pd.DataFrame | None = None
        self.model_metrics_raw: dict[str, Any] | None = None
        self.model_metrics: dict[str, Any] | None = None
        self.kpi: dict[str, Any] | None = None
        self.assumptions: dict[str, Any] | None = None
        self.feedback_df: pd.DataFrame | None = None

    def load_all(self) -> None:
        preferred, fallback = DATA_FILES["scores"]
        self.scores_df = read_csv_df(preferred, fallback)
        self.segment_df = read_csv_df(*DATA_FILES["segments"])
        self.feature_df = read_csv_df(*DATA_FILES["features"])
        self.model_metrics_raw = read_json_obj(*DATA_FILES["metrics"])
        self.model_metrics = normalize_model_metrics(self.model_metrics_raw)

        detail_path = DATA_DIR / DATA_FILES["detail"][0]
        self.detail_df = (
            pd.read_csv(detail_path, low_memory=False) if detail_path.exists() else None
        )

        banks_path = DATA_DIR / DATA_FILES["banks"][0]
        self.banks_df = pd.read_csv(banks_path) if banks_path.exists() else None

        products_path = DATA_DIR / DATA_FILES["products"][0]
        self.products_df = pd.read_csv(products_path) if products_path.exists() else None

        cluster_path = DATA_DIR / DATA_FILES["cluster"][0]
        self.cluster_df = pd.read_csv(cluster_path) if cluster_path.exists() else None

        self.kpi = read_json_optional(DATA_FILES["kpi"][0])
        self.assumptions = read_json_optional(DATA_FILES["assumptions"][0])

        feedback_path = DATA_DIR / DATA_FILES["feedback"][0]
        if feedback_path.exists():
            self.feedback_df = pd.read_csv(feedback_path)
        else:
            self.feedback_df = pd.DataFrame(
                columns=["card_id", "user_role", "feedback_status", "comment", "date"]
            )

        load_scores_df.cache_clear()

    def refresh_scores(self) -> None:
        preferred, fallback = DATA_FILES["scores"]
        self.scores_df = read_csv_df(preferred, fallback)
        load_scores_df.cache_clear()

    def refresh_feedback(self) -> None:
        path = feedback_write_path()
        self.feedback_df = pd.read_csv(path) if path.exists() else pd.DataFrame(
            columns=["card_id", "user_role", "feedback_status", "comment", "date"]
        )


store = DataStore()


def _scores_dataframe() -> pd.DataFrame:
    if store.scores_df is not None:
        return store.scores_df
    return load_scores_df()


def _detail_dataframe() -> pd.DataFrame | None:
    return store.detail_df


def leads_role_is_aggregated(role: str | None) -> bool:
    if not role:
        return True
    normalized = role.strip().lower()
    if normalized in AGGREGATED_LEAD_ROLES:
        return True
    if normalized in FULL_LEAD_ROLES:
        return False
    return normalized not in FULL_LEAD_ROLES


def filter_leads_df(
    df: pd.DataFrame,
    *,
    bank: str | None = None,
    segment: str | None = None,
    feedback: str | None = None,
    search: str | None = None,
) -> pd.DataFrame:
    filtered = _ensure_risk_tiers(df.copy())

    if bank:
        filtered = filtered[filtered["bank_name"].astype(str).str.lower() == bank.lower()]

    if feedback:
        filtered = filtered[
            filtered["feedback_status"].astype(str).str.lower() == feedback.lower()
        ]

    if segment:
        seg_lower = segment.lower()
        mask = (
            filtered["risk_segment"].astype(str).str.lower() == seg_lower
        ) | (filtered["opportunity_segment"].astype(str).str.lower() == seg_lower)
        filtered = filtered[mask]

    if search and "card_id" in filtered.columns:
        q = search.lower()
        filtered = filtered[
            filtered["card_id"].astype(str).str.lower().str.contains(q, na=False)
        ]

    return filtered


def sort_leads_df(df: pd.DataFrame, sort: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    key = (sort or "score_desc").strip().lower()
    if key in ("expected_value_asc", "value_asc"):
        return df.sort_values("expected_value_kzt", ascending=True)
    if key in ("expected_value_desc", "value_desc"):
        return df.sort_values("expected_value_kzt", ascending=False)
    if key == "bank":
        return df.sort_values(
            ["bank_name", "commercial_activity_score"],
            ascending=[True, False],
        )
    return df.sort_values("commercial_activity_score", ascending=False)


def aggregate_leads(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "view": "aggregated",
            "total_leads": 0,
            "average_score": 0.0,
            "total_expected_value_kzt": 0.0,
            "by_bank": [],
            "by_segment": [],
        }

    by_bank = (
        df.groupby("bank_name", dropna=False)
        .agg(
            lead_count=("card_id", "count"),
            average_score=("commercial_activity_score", "mean"),
            total_expected_value_kzt=("expected_value_kzt", "sum"),
        )
        .reset_index()
        .sort_values("lead_count", ascending=False)
    )
    by_segment = (
        df.groupby("risk_segment", dropna=False)
        .agg(
            lead_count=("card_id", "count"),
            average_score=("commercial_activity_score", "mean"),
            total_expected_value_kzt=("expected_value_kzt", "sum"),
        )
        .reset_index()
        .sort_values("lead_count", ascending=False)
    )

    return {
        "view": "aggregated",
        "total_leads": int(len(df)),
        "average_score": float(df["commercial_activity_score"].mean()),
        "total_expected_value_kzt": float(df["expected_value_kzt"].sum()),
        "by_bank": json.loads(by_bank.to_json(orient="records")),
        "by_segment": json.loads(by_segment.to_json(orient="records")),
    }


def query_leads(
    *,
    role: str | None = None,
    bank: str | None = None,
    segment: str | None = None,
    feedback: str | None = None,
    search: str | None = None,
    sort: str | None = None,
    limit: int | None = 500,
    offset: int = 0,
) -> dict[str, Any]:
    df = filter_leads_df(
        _scores_dataframe(),
        bank=bank,
        segment=segment,
        feedback=feedback,
        search=search,
    )
    df = sort_leads_df(df, sort)

    if leads_role_is_aggregated(role):
        return aggregate_leads(df)

    total = int(len(df))
    if offset:
        df = df.iloc[offset:]
    if limit is not None:
        df = df.head(limit)

    leads = [
        normalize_cardholder_record(r)
        for r in json.loads(df.to_json(orient="records"))
    ]
    return {"leads": leads, "total": total}


def join_cardholder(card_id: str) -> dict[str, Any] | None:
    scores = _scores_dataframe()
    score_match = scores[scores["card_id"] == card_id]
    detail_df = _detail_dataframe()
    detail_match = (
        detail_df[detail_df["card_id"] == card_id]
        if detail_df is not None
        else pd.DataFrame()
    )

    if score_match.empty and detail_match.empty:
        return None

    merged: dict[str, Any] = {}
    if not score_match.empty:
        merged.update(json.loads(score_match.to_json(orient="records"))[0])
    if not detail_match.empty:
        merged.update(json.loads(detail_match.to_json(orient="records"))[0])

    return normalize_cardholder_record(merged)


def audit_log_records_sorted() -> list[dict[str, Any]]:
    path = audit_write_path()
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    sort_col = "timestamp" if "timestamp" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    return [strip_sensitive_columns(r) for r in json.loads(df.to_json(orient="records"))]


def feedback_log_records_sorted() -> list[dict[str, Any]]:
    df = store.feedback_df
    if df is None or df.empty:
        path = feedback_write_path()
        if not path.exists():
            return []
        df = pd.read_csv(path)
    else:
        df = df.copy()

    if "date" in df.columns:
        df = df.sort_values("date", ascending=False)
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    return [strip_sensitive_columns(r) for r in json.loads(df.to_json(orient="records"))]
