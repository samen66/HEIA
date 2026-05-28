from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from business_impact import calculate_business_impact

DATA_DIR = Path(__file__).resolve().parent / "data"

DEFAULT_CONVERSION_PCT = 10.0
DEFAULT_AVG_REVENUE_KZT = 50_000.0
DEFAULT_CAMPAIGN_COST_KZT = 5_000_000.0

def _scores_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "demo_cardholder_scores.csv")


def _segments_df() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "demo_segment_summary.csv")


def _extract_percent(question: str, default: float | None = None) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", question)
    if match:
        return float(match.group(1))
    match = re.search(
        r"(?:conversion|rate|roi)\s*(?:of|at|to|is)?\s*(\d+(?:\.\d+)?)",
        question,
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    return default


def _high_opportunity_count(scores: pd.DataFrame) -> int:
    return int((scores["risk_segment"] == "High").sum())


def answer_agent_question(question: str) -> dict[str, Any]:
    q = question.lower().strip()
    scores = _scores_df()
    high_count = _high_opportunity_count(scores)

    if "roi" in q:
        conversion_pct = _extract_percent(question, DEFAULT_CONVERSION_PCT) or DEFAULT_CONVERSION_PCT
        impact = calculate_business_impact(
            high_opportunity_customers=high_count,
            conversion_rate_pct=conversion_pct,
            avg_annual_revenue_kzt=DEFAULT_AVG_REVENUE_KZT,
            campaign_cost_kzt=DEFAULT_CAMPAIGN_COST_KZT,
        )
        answer = (
            f"At a {conversion_pct:.0f}% conversion rate on {high_count:,} high-opportunity "
            f"customers, estimated gross revenue is "
            f"₸{impact['estimated_gross_revenue_kzt']:,.0f} with net impact "
            f"₸{impact['net_business_impact_kzt']:,.0f}, yielding ROI of "
            f"{impact['roi_pct']:.1f}%."
        )
        return {
            "answer": answer,
            "supporting_data": {
                "conversion_rate_pct": conversion_pct,
                "high_opportunity_customers": high_count,
                "roi_pct": impact["roi_pct"],
                "net_business_impact_kzt": impact["net_business_impact_kzt"],
            },
        }

    if "revenue" in q or "conversion" in q:
        conversion_pct = _extract_percent(question, DEFAULT_CONVERSION_PCT) or DEFAULT_CONVERSION_PCT
        impact = calculate_business_impact(
            high_opportunity_customers=high_count,
            conversion_rate_pct=conversion_pct,
            avg_annual_revenue_kzt=DEFAULT_AVG_REVENUE_KZT,
            campaign_cost_kzt=DEFAULT_CAMPAIGN_COST_KZT,
        )
        answer = (
            f"With {high_count:,} high-opportunity customers and a {conversion_pct:.0f}% "
            f"conversion rate, we expect roughly {impact['converted_customers']:.1f} "
            f"conversions and estimated gross revenue of "
            f"₸{impact['estimated_gross_revenue_kzt']:,.0f} per year "
            f"(avg ₸{DEFAULT_AVG_REVENUE_KZT:,.0f} per customer)."
        )
        return {
            "answer": answer,
            "supporting_data": {
                "conversion_rate_pct": conversion_pct,
                "high_opportunity_customers": high_count,
                "converted_customers": round(impact["converted_customers"], 1),
                "estimated_gross_revenue_kzt": impact["estimated_gross_revenue_kzt"],
            },
        }

    if "how many" in q and "high" in q:
        segments = _segments_df()
        high_row = segments[segments["risk_segment"] == "High"].iloc[0]
        answer = (
            f"There are {high_count} high-opportunity customers in the scored portfolio "
            f"(avg commercial activity score {high_row['average_score']:.2f}, "
            f"estimated segment opportunity ₸{high_row['estimated_opportunity_value']:,.0f})."
        )
        return {
            "answer": answer,
            "supporting_data": {
                "high_opportunity_customers": high_count,
                "average_score": round(float(high_row["average_score"]), 2),
                "estimated_opportunity_value": int(high_row["estimated_opportunity_value"]),
            },
        }

    if "bank" in q:
        high_scores = scores[scores["risk_segment"] == "High"]
        bank_counts = high_scores["bank_name"].value_counts()
        top_bank = bank_counts.index[0]
        top_count = int(bank_counts.iloc[0])
        answer = (
            f"{top_bank} has the most hidden entrepreneurs in the High segment, "
            f"with {top_count} high-opportunity cardholders "
            f"({top_count / high_count:.0%} of the High segment)."
        )
        return {
            "answer": answer,
            "supporting_data": {
                "top_bank": top_bank,
                "high_opportunity_at_bank": top_count,
                "high_segment_total": high_count,
            },
        }

    if "reason" in q or "flagged" in q:
        reason_cols = ["top_reason_1", "top_reason_2", "top_reason_3"]
        all_reasons: list[str] = []
        for col in reason_cols:
            all_reasons.extend(scores[col].dropna().astype(str).tolist())
        top_three = Counter(all_reasons).most_common(3)
        lines = [f"{i + 1}. {reason} ({count} cardholders)" for i, (reason, count) in enumerate(top_three)]
        answer = (
            "The top reasons customers are flagged for hidden entrepreneur signals are:\n"
            + "\n".join(lines)
        )
        supporting: dict[str, Any] = {}
        for i, (reason, count) in enumerate(top_three, start=1):
            supporting[f"reason_{i}"] = reason
            supporting[f"reason_{i}_count"] = count
        return {"answer": answer, "supporting_data": supporting}

    if "product" in q:
        segments = _segments_df()
        lines: list[str] = []
        supporting: dict[str, Any] = {}
        for _, row in segments.iterrows():
            seg = row["risk_segment"]
            product = row["recommended_product"]
            count = int((scores["risk_segment"] == seg).sum())
            lines.append(f"• {seg}: {product} ({count} cardholders)")
            supporting[f"{seg.lower()}_product"] = product
            supporting[f"{seg.lower()}_cardholders"] = count
        if "high" in q:
            high_product = segments[segments["risk_segment"] == "High"]["recommended_product"].iloc[0]
            answer = (
                f"For the High segment, recommend {high_product}. "
                f"Full segment playbook:\n" + "\n".join(lines)
            )
        else:
            answer = "Recommended products by opportunity segment:\n" + "\n".join(lines)
        return {"answer": answer, "supporting_data": supporting}

    return {
        "answer": (
            "I can answer questions about opportunity segments, revenue estimates, "
            "model performance, and recommended actions. Please try one of the suggested questions."
        ),
        "supporting_data": {"high_opportunity_customers": high_count},
    }
