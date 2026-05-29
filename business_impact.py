from __future__ import annotations

from typing import Any


def calculate_business_impact(
    high_opportunity_customers: int,
    conversion_rate_pct: float,
    avg_annual_revenue_kzt: float,
    campaign_cost_kzt: float,
) -> dict[str, Any]:
    rate = conversion_rate_pct / 100.0
    converted_customers = high_opportunity_customers * rate
    estimated_gross_revenue_kzt = converted_customers * avg_annual_revenue_kzt
    net_business_impact_kzt = estimated_gross_revenue_kzt - campaign_cost_kzt
    roi_pct = (
        (net_business_impact_kzt / campaign_cost_kzt) * 100
        if campaign_cost_kzt > 0
        else 0.0
    )

    return {
        "high_opportunity_customers": high_opportunity_customers,
        "conversion_rate_pct": conversion_rate_pct,
        "avg_annual_revenue_kzt": avg_annual_revenue_kzt,
        "campaign_cost_kzt": campaign_cost_kzt,
        "converted_customers": round(converted_customers, 2),
        "estimated_gross_revenue_kzt": round(estimated_gross_revenue_kzt, 2),
        "net_business_impact_kzt": round(net_business_impact_kzt, 2),
        "roi_pct": round(roi_pct, 2),
    }


def calculate_business_impact_v2(
    *,
    leads_count: int,
    conversion_rate: float,
    avg_revenue_per_customer: float,
    campaign_cost: float,
) -> dict[str, Any]:
    """conversion_rate is a fraction in [0, 1] (e.g. 0.12 for 12%)."""
    converted_customers = leads_count * conversion_rate
    gross_revenue = converted_customers * avg_revenue_per_customer
    net_impact = gross_revenue - campaign_cost
    roi = (net_impact / campaign_cost) if campaign_cost > 0 else 0.0

    return {
        "leads_count": leads_count,
        "conversion_rate": conversion_rate,
        "avg_revenue_per_customer": avg_revenue_per_customer,
        "campaign_cost": campaign_cost,
        "converted_customers": round(converted_customers, 2),
        "gross_revenue": round(gross_revenue, 2),
        "net_impact": round(net_impact, 2),
        "roi": round(roi, 4),
    }
