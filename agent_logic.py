from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import pandas as pd

from business_impact import calculate_business_impact
from data_loader import (
    format_kpi_summary_text,
    feedback_status_summary,
    load_scores_df,
    load_segments_df,
    normalize_segment_records,
    read_json_optional,
    risk_segment_for,
    store,
)

DEFAULT_CONVERSION_PCT = 10.0
DEFAULT_AVG_REVENUE_KZT = 50_000.0
DEFAULT_CAMPAIGN_COST_KZT = 5_000_000.0

DIRECTOR_BLOCKED_MESSAGE = "Your role allows aggregated insights only."

_CARDHOLDER_DETAIL_PATTERNS = (
    r"\bcard[_\s-]?id\b",
    r"\bcardholder\s+details?\b",
    r"\bindividual\s+cardholder\b",
    r"\bspecific\s+cardholder\b",
    r"\bshow\s+me\s+.*\bcard\b",
    r"конкретн\w*\s+карт",
    r"отдельн\w*\s+карт",
    r"детал\w*\s+картодерж",
    r"индивидуальн\w*\s+карт",
)


def director_requests_cardholder_detail(role: str, question: str) -> bool:
    if role.strip().lower() != "director":
        return False
    q = question.lower()
    return any(re.search(pat, q, re.IGNORECASE) for pat in _CARDHOLDER_DETAIL_PATTERNS)


def detect_intent(question: str) -> str | None:
    q = question.lower().strip()

    if any(
        w in q
        for w in (
            "false positive",
            "ложнополож",
            "ложн.*полож",
            "feedback case",
            "feedback status",
            "show false",
        )
    ):
        return "feedback_cases"

    if any(
        w in q
        for w in (
            "limitation",
            "limitations",
            "proxy validation",
            "ограничен",
            "ограничени",
            "model limit",
        )
    ):
        return "limitations"

    if any(
        w in q
        for w in (
            "feature",
            "features matter",
            "feature importance",
            "признак",
            "сигнал",
            "фактор",
        )
    ) and not ("product" in q or "продукт" in q):
        return "feature_importance"

    if any(
        w in q
        for w in (
            "roc",
            "auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "model metric",
            "model performance",
            "метрик модел",
        )
    ):
        return "model_metrics"

    if any(
        w in q
        for w in (
            "roi",
            "conversion",
            "конверси",
            "revenue",
            "выручк",
            "доход",
            "если конверсия",
            "business impact",
            "бизнес-эффект",
        )
    ):
        return "business_impact"

    if any(
        w in q
        for w in (
            "product",
            "продукт",
            "recommend",
            "рекоменд",
            "offer",
            "предложен",
        )
    ):
        return "recommended_products"

    if any(w in q for w in ("bank", "банк", "issuer")):
        return "top_banks"

    if any(
        w in q
        for w in (
            "total opportunity",
            "общий потенциал",
            "скрытых предпринимател",
            "hidden entrepreneur",
            "portfolio summary",
            "kpi",
            "сколько кандидат",
        )
    ):
        return "total_opportunity"

    return None


def _extract_percent(question: str, default: float | None = None) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", question)
    if match:
        return float(match.group(1))
    match = re.search(
        r"(?:conversion|rate|roi|конверси)\w*\s*(?:of|at|to|is|до|выраст\w*)?\s*(\d+(?:\.\d+)?)",
        question,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return default


def _scores_df() -> pd.DataFrame:
    df = (store.scores_df if store.scores_df is not None else load_scores_df()).copy()
    if "risk_segment" not in df.columns or df["risk_segment"].isin(["High", "Medium", "Low"]).sum() == 0:
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


def _high_opportunity_count(scores: pd.DataFrame) -> int:
    return int((scores["risk_segment"] == "High").sum())


def _feature_columns(df: pd.DataFrame) -> tuple[str, str]:
    for feat, imp in (
        ("feature_name", "importance_value"),
        ("feature", "importance"),
    ):
        if feat in df.columns and imp in df.columns:
            return feat, imp
    cols = list(df.columns)
    return cols[0], cols[1] if len(cols) > 1 else cols[0]


def _segments_list() -> list[dict[str, Any]]:
    assumptions = read_json_optional("business_impact_assumptions.json")
    seg_df = store.segment_df if store.segment_df is not None else load_segments_df()
    return normalize_segment_records(seg_df, assumptions)


def _business_impact_block(question: str) -> tuple[str, dict[str, Any]]:
    scores = _scores_df()
    high_count = _high_opportunity_count(scores)
    conversion_pct = _extract_percent(question, DEFAULT_CONVERSION_PCT) or DEFAULT_CONVERSION_PCT
    impact = calculate_business_impact(
        high_opportunity_customers=high_count,
        conversion_rate_pct=conversion_pct,
        avg_annual_revenue_kzt=DEFAULT_AVG_REVENUE_KZT,
        campaign_cost_kzt=DEFAULT_CAMPAIGN_COST_KZT,
    )
    text = (
        f"High-opportunity customers: {high_count:,}\n"
        f"Conversion rate: {conversion_pct:.1f}%\n"
        f"Converted customers: {impact['converted_customers']:.1f}\n"
        f"Estimated gross revenue: ₸{impact['estimated_gross_revenue_kzt']:,.0f}\n"
        f"Campaign cost: ₸{impact['campaign_cost_kzt']:,.0f}\n"
        f"Net business impact: ₸{impact['net_business_impact_kzt']:,.0f}\n"
        f"ROI: {impact['roi_pct']:.1f}%"
    )
    supporting = {
        "conversion_rate_pct": conversion_pct,
        "high_opportunity_customers": high_count,
        "roi_pct": impact["roi_pct"],
        "net_business_impact_kzt": impact["net_business_impact_kzt"],
        "estimated_gross_revenue_kzt": impact["estimated_gross_revenue_kzt"],
    }
    return text, supporting


def build_intent_focus_block(intent: str | None, question: str) -> str:
    if not intent:
        return ""

    if intent == "total_opportunity":
        return format_kpi_summary_text(store.kpi)

    if intent == "top_banks" and store.banks_df is not None:
        return store.banks_df.head(10).to_string()

    if intent == "recommended_products" and store.products_df is not None:
        top = store.products_df.sort_values(
            "number_of_cardholders", ascending=False
        ).head(5)
        return top.to_string()

    if intent == "business_impact":
        block, _ = _business_impact_block(question)
        return block

    if intent == "model_metrics" and store.model_metrics:
        return json.dumps(store.model_metrics, indent=2)

    if intent == "feature_importance" and store.feature_df is not None:
        return store.feature_df.head(10).to_string()

    if intent == "limitations" and store.model_metrics:
        note = store.model_metrics.get("important_note", "")
        proxy = store.model_metrics.get("proxy_validation", [])
        return f"{note}\n\nProxy validation:\n{json.dumps(proxy, indent=2)}"

    if intent == "feedback_cases":
        return feedback_status_summary(store.feedback_df)

    return ""


def intent_supporting_data(intent: str | None, question: str) -> dict[str, Any]:
    if not intent:
        return {}

    if intent == "business_impact":
        _, supporting = _business_impact_block(question)
        return supporting

    if intent == "total_opportunity" and store.kpi:
        return {
            k: v
            for k, v in store.kpi.items()
            if isinstance(v, (int, float))
        }

    if intent == "top_banks" and store.banks_df is not None and not store.banks_df.empty:
        top = store.banks_df.iloc[0]
        return {
            "top_bank": str(top.get("bank_name", "")),
            "estimated_opportunity_value": float(top.get("estimated_opportunity_value", 0)),
            "high_priority_count": int(top.get("high_priority_count", 0)),
        }

    if intent == "recommended_products" and store.products_df is not None:
        top = store.products_df.sort_values("number_of_cardholders", ascending=False).iloc[0]
        return {
            "top_recommended_action": str(top.get("recommended_action", "")),
            "cardholders": int(top.get("number_of_cardholders", 0)),
        }

    if intent == "model_metrics" and store.model_metrics:
        return {
            k: store.model_metrics[k]
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc")
            if k in store.model_metrics
        }

    if intent == "feature_importance" and store.feature_df is not None:
        feat_col, imp_col = _feature_columns(store.feature_df)
        row = store.feature_df.iloc[0]
        return {
            "top_feature": str(row[feat_col]),
            "importance": float(row[imp_col]),
        }

    if intent == "feedback_cases" and store.feedback_df is not None and not store.feedback_df.empty:
        fp = int(
            (store.feedback_df["feedback_status"].astype(str) == "False positive").sum()
        )
        return {"false_positive_cases": fp, "total_feedback_rows": len(store.feedback_df)}

    return {"intent": intent}


def answer_agent_question(
    question: str,
    *,
    role: str = "guest",
    intent: str | None = None,
) -> dict[str, Any]:
    if director_requests_cardholder_detail(role, question):
        return {"answer": DIRECTOR_BLOCKED_MESSAGE, "supporting_data": {"blocked": True}}

    resolved_intent = intent or detect_intent(question)
    if resolved_intent:
        block = build_intent_focus_block(resolved_intent, question)
        supporting = intent_supporting_data(resolved_intent, question)
        if resolved_intent == "total_opportunity" and store.kpi:
            kpi = store.kpi
            opp = float(kpi.get("estimated_total_opportunity_kzt", 0))
            total = int(kpi.get("total_scored_consumers", 0))
            answer = (
                f"The portfolio includes {total:,} scored consumers with an estimated "
                f"total hidden-entrepreneur opportunity of ₸{opp:,.0f} "
                f"(average score {float(kpi.get('average_score', 0)):.4f})."
            )
            return {"answer": answer, "supporting_data": supporting}

        if resolved_intent == "top_banks" and store.banks_df is not None and not store.banks_df.empty:
            top = store.banks_df.iloc[0]
            bank = str(top["bank_name"])
            opp = float(top["estimated_opportunity_value"])
            high = int(top.get("high_priority_count", 0))
            answer = (
                f"{bank} has the largest estimated opportunity at ₸{opp:,.0f} "
                f"with {high:,} high-priority cardholders in the scored portfolio."
            )
            return {"answer": answer, "supporting_data": supporting}

        if resolved_intent == "recommended_products" and store.products_df is not None:
            top = store.products_df.sort_values("number_of_cardholders", ascending=False).iloc[0]
            action = str(top["recommended_action"])
            count = int(top["number_of_cardholders"])
            answer = (
                f"The most frequently recommended action is “{action}” "
                f"for {count:,} cardholders."
            )
            return {"answer": answer, "supporting_data": supporting}

        if resolved_intent == "business_impact":
            _, supporting = _business_impact_block(question)
            conversion_pct = supporting["conversion_rate_pct"]
            answer = (
                f"At {conversion_pct:.0f}% conversion on "
                f"{supporting['high_opportunity_customers']:,} high-opportunity customers, "
                f"estimated gross revenue is ₸{supporting['estimated_gross_revenue_kzt']:,.0f} "
                f"with ROI {supporting['roi_pct']:.1f}%."
            )
            return {"answer": answer, "supporting_data": supporting}

        if resolved_intent == "model_metrics" and store.model_metrics:
            m = store.model_metrics
            answer = (
                f"Model {m.get('model_name', 'HEIS')}: accuracy {m.get('accuracy', 0):.1%}, "
                f"precision {m.get('precision', 0):.1%}, recall {m.get('recall', 0):.1%}, "
                f"ROC-AUC {m.get('roc_auc', 0):.3f}."
            )
            return {"answer": answer, "supporting_data": supporting}

        if resolved_intent == "feature_importance" and store.feature_df is not None:
            feat_col, imp_col = _feature_columns(store.feature_df)
            lines = [
                f"{i + 1}. {row[feat_col]} ({float(row[imp_col]):.4f})"
                for i, (_, row) in enumerate(store.feature_df.head(5).iterrows())
            ]
            answer = "Top predictive feature signals:\n" + "\n".join(lines)
            return {"answer": answer, "supporting_data": supporting}

        if resolved_intent == "limitations" and store.model_metrics:
            note = store.model_metrics.get(
                "important_note",
                "Proxy validation only — consumer cards are an unlabeled candidate pool.",
            )
            return {"answer": note, "supporting_data": supporting}

        if resolved_intent == "feedback_cases":
            if store.feedback_df is None or store.feedback_df.empty:
                return {
                    "answer": "No feedback cases logged yet in the demo portfolio.",
                    "supporting_data": supporting,
                }
            counts = store.feedback_df["feedback_status"].value_counts()
            lines = [f"• {status}: {int(cnt)}" for status, cnt in counts.items()]
            answer = "Feedback status summary:\n" + "\n".join(lines)
            return {"answer": answer, "supporting_data": supporting}

        if block:
            return {
                "answer": f"Based on portfolio data:\n{block}",
                "supporting_data": supporting,
            }

    return _legacy_answer(question)


def _legacy_answer(question: str) -> dict[str, Any]:
    """Fallback keyword routing when intent is not recognized."""
    q = question.lower().strip()
    scores = _scores_df()
    high_count = _high_opportunity_count(scores)

    if "roi" in q or ("conversion" in q and "%" in question):
        _, supporting = _business_impact_block(question)
        conversion_pct = supporting["conversion_rate_pct"]
        answer = (
            f"At a {conversion_pct:.0f}% conversion rate on {high_count:,} high-opportunity "
            f"customers, estimated gross revenue is "
            f"₸{supporting['estimated_gross_revenue_kzt']:,.0f} with ROI "
            f"{supporting['roi_pct']:.1f}%."
        )
        return {"answer": answer, "supporting_data": supporting}

    if "how many" in q and "high" in q:
        segments = _segments_list()
        high_row = next((s for s in segments if s["risk_segment"] == "High"), None)
        if high_row:
            answer = (
                f"There are {high_count:,} high-opportunity customers "
                f"(avg score {high_row['average_score']:.2f})."
            )
            return {
                "answer": answer,
                "supporting_data": {"high_opportunity_customers": high_count},
            }
        return {
            "answer": f"There are {high_count:,} high-opportunity customers.",
            "supporting_data": {"high_opportunity_customers": high_count},
        }

    if "reason" in q or "flagged" in q:
        reason_cols = ["top_reason_1", "top_reason_2", "top_reason_3"]
        all_reasons: list[str] = []
        for col in reason_cols:
            if col in scores.columns:
                all_reasons.extend(scores[col].dropna().astype(str).tolist())
        top_three = Counter(all_reasons).most_common(3)
        lines = [
            f"{i + 1}. {reason} ({count:,} cardholders)"
            for i, (reason, count) in enumerate(top_three)
        ]
        answer = "Top flagging reasons:\n" + "\n".join(lines)
        return {"answer": answer, "supporting_data": {}}

    return {
        "answer": (
            "I can answer questions about portfolio opportunity, banks, products, "
            "model metrics, features, business impact scenarios, and feedback cases. "
            "Try one of the suggested questions."
        ),
        "supporting_data": {"high_opportunity_customers": high_count},
    }
