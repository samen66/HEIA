"""Generate HEIA demo data files. Run once: python scripts/generate_demo_data.py"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

BANKS = [
    "Halyk Bank",
    "Kaspi Bank",
    "ForteBank",
    "Jusan Bank",
    "Bank CenterCredit",
    "Eurasian Bank",
    "Home Credit Bank",
]
CARD_TIERS = ["Standard", "Gold", "Platinum", "World", "Business"]
RISK_SEGMENTS = ["High", "Medium", "Low"]
OPPORTUNITY_SEGMENTS = [
    "SME Lending",
    "Merchant Acquiring",
    "Working Capital",
    "Trade Finance",
    "Treasury Services",
    "Payroll Solutions",
]
ACTIONS = [
    "Offer SME credit line",
    "Invite to merchant program",
    "Schedule relationship call",
    "Cross-sell working capital",
    "Enable trade finance package",
    "Propose payroll bundle",
    "Monitor — nurture in 90 days",
]
REASONS = [
    "High cross-border payment velocity",
    "Growing monthly turnover vs peer median",
    "Frequent B2B supplier payments",
    "Seasonal revenue spike detected",
    "Low utilization of existing credit",
    "Strong repayment history on retail products",
    "Multiple POS terminals registered",
    "E-commerce platform integrations active",
    "Cash flow volatility within acceptable band",
    "Industry sector growth above national average",
]
FEEDBACK_STATUSES = ["Pending", "Accepted", "Rejected", "Deferred"]
USER_ROLES = ["Relationship Manager", "Credit Analyst", "Portfolio Manager", "Branch Manager"]


def generate_cardholder_scores(n: int = 200) -> None:
    rows = []
    for i in range(1, n + 1):
        risk = random.choices(RISK_SEGMENTS, weights=[0.25, 0.45, 0.30])[0]
        score = round(
            random.uniform(0.55, 0.95)
            if risk == "High"
            else random.uniform(0.35, 0.75)
            if risk == "Medium"
            else random.uniform(0.15, 0.55),
            3,
        )
        reasons = random.sample(REASONS, 3)
        ev = round(score * random.uniform(800_000, 4_500_000), 0)
        rows.append(
            {
                "card_id": f"MC-KZ-{i:05d}",
                "bank_name": random.choice(BANKS),
                "card_tier": random.choice(CARD_TIERS),
                "commercial_activity_score": score,
                "risk_segment": risk,
                "opportunity_segment": random.choice(OPPORTUNITY_SEGMENTS),
                "recommended_action": random.choice(ACTIONS),
                "confidence_level": round(random.uniform(0.72, 0.98), 2),
                "top_reason_1": reasons[0],
                "top_reason_2": reasons[1],
                "top_reason_3": reasons[2],
                "expected_value_kzt": int(ev),
                "feedback_status": random.choice(FEEDBACK_STATUSES),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "demo_cardholder_scores.csv", index=False)


def generate_model_metrics() -> None:
    payload = {
        "model_name": "LightGBM",
        "model_version": "heia-v1.2.0",
        "trained_at": "2025-11-18T14:30:00Z",
        "accuracy": 0.91,
        "precision": 0.84,
        "recall": 0.79,
        "f1": 0.81,
        "roc_auc": 0.93,
        "training_samples": 18420,
        "validation_samples": 4605,
        "positive_class_rate": 0.22,
        "notes": "Commercial activity classifier for Kazakhstan Mastercard portfolio",
    }
    (DATA_DIR / "demo_model_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def generate_feature_importance() -> None:
    features = [
        ("monthly_turnover_kzt", 0.187, "Average monthly card spend volume in KZT"),
        ("b2b_payment_ratio", 0.156, "Share of payments to registered business merchants"),
        ("cross_border_txn_count", 0.134, "International transaction frequency (90d)"),
        ("pos_terminal_count", 0.112, "Number of linked point-of-sale terminals"),
        ("credit_utilization_rate", 0.098, "Utilization of existing credit lines"),
        ("cash_withdrawal_ratio", 0.087, "Cash advance share of total volume"),
        ("merchant_category_diversity", 0.079, "Entropy of MCC codes used"),
        ("account_tenure_months", 0.071, "Months since card activation"),
        ("weekend_activity_ratio", 0.042, "Weekend vs weekday spend pattern"),
        ("delinquency_flag_12m", 0.034, "Any 30+ DPD events in trailing 12 months"),
    ]
    pd.DataFrame(
        features, columns=["feature_name", "importance_value", "business_meaning"]
    ).to_csv(DATA_DIR / "demo_feature_importance.csv", index=False)


def generate_segment_summary() -> None:
    segments = [
        {
            "risk_segment": "High",
            "number_of_cardholders": 52,
            "average_score": 0.78,
            "recommended_product": "SME Credit Line + Merchant Acquiring",
            "estimated_conversion_rate": 0.34,
            "estimated_opportunity_value": 186_400_000,
        },
        {
            "risk_segment": "Medium",
            "number_of_cardholders": 89,
            "average_score": 0.54,
            "recommended_product": "Working Capital Facility",
            "estimated_conversion_rate": 0.21,
            "estimated_opportunity_value": 124_800_000,
        },
        {
            "risk_segment": "Low",
            "number_of_cardholders": 59,
            "average_score": 0.31,
            "recommended_product": "Relationship Nurture Program",
            "estimated_conversion_rate": 0.09,
            "estimated_opportunity_value": 42_600_000,
        },
    ]
    pd.DataFrame(segments).to_csv(DATA_DIR / "demo_segment_summary.csv", index=False)


def generate_feedback_log(n: int = 20) -> None:
    rows = []
    base = datetime(2025, 10, 1)
    for i in range(1, n + 1):
        rows.append(
            {
                "card_id": f"MC-KZ-{random.randint(1, 200):05d}",
                "user_role": random.choice(USER_ROLES),
                "feedback_status": random.choice(["Accepted", "Rejected", "Deferred"]),
                "comment": random.choice(
                    [
                        "Strong fit for SME lending — proceed with offer",
                        "Client already has competing product at another bank",
                        "Need additional KYC documentation before outreach",
                        "Timing not right — revisit Q1 2026",
                        "Confirmed hidden entrepreneur profile in follow-up call",
                        "Risk appetite too high for current portfolio mix",
                        "Approved for merchant acquiring pilot program",
                    ]
                ),
                "date": (base + timedelta(days=random.randint(0, 120))).strftime(
                    "%Y-%m-%d"
                ),
            }
        )
    pd.DataFrame(rows).to_csv(DATA_DIR / "demo_feedback_log.csv", index=False)


if __name__ == "__main__":
    generate_cardholder_scores()
    generate_model_metrics()
    generate_feature_importance()
    generate_segment_summary()
    generate_feedback_log()
    print(f"Demo data written to {DATA_DIR}")
