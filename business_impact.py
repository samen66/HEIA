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
