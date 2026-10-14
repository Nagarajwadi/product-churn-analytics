"""Scenario-based retention campaign economics.

All outputs are estimates based on assumptions, not measured campaign effects.
"""

import math


def calculate_retention_roi(
    audience_size: int,
    campaign_cost_per_user: float,
    retention_uplift_pct: float,
    contribution_per_retained_user: float,
) -> dict:
    """Calculate estimated retention campaign economics."""
    if isinstance(audience_size, bool) or not isinstance(audience_size, int):
        raise ValueError("audience_size must be a non-negative integer.")
    if audience_size < 0:
        raise ValueError("audience_size must be a non-negative integer.")

    for name, value in (
        ("campaign_cost_per_user", campaign_cost_per_user),
        ("retention_uplift_pct", retention_uplift_pct),
        ("contribution_per_retained_user", contribution_per_retained_user),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative.")

    if retention_uplift_pct > 100:
        raise ValueError("retention_uplift_pct must not exceed 100.")

    campaign_cost = audience_size * campaign_cost_per_user
    incremental_retained_users = audience_size * retention_uplift_pct / 100
    incremental_contribution = (
        incremental_retained_users * contribution_per_retained_user
    )
    net_benefit = incremental_contribution - campaign_cost
    roi_pct = (
        net_benefit / campaign_cost * 100
        if campaign_cost > 0
        else None
    )
    break_even_uplift_pct = (
        campaign_cost
        / (audience_size * contribution_per_retained_user)
        * 100
        if audience_size > 0 and contribution_per_retained_user > 0
        else None
    )

    return {
        "audience_size": audience_size,
        "campaign_cost": campaign_cost,
        "incremental_retained_users": incremental_retained_users,
        "incremental_contribution": incremental_contribution,
        "net_benefit": net_benefit,
        "roi_pct": roi_pct,
        "break_even_uplift_pct": break_even_uplift_pct,
    }


def compare_retention_scenarios(
    audience_size: int,
    campaign_cost_per_user: float,
    retention_uplifts_pct: list[float],
    contribution_per_retained_user: float,
) -> list[dict]:
    """Compare estimated outcomes across retention-uplift assumptions."""
    if not retention_uplifts_pct:
        raise ValueError("retention_uplifts_pct must not be empty.")

    return [
        {
            "retention_uplift_pct": uplift,
            **calculate_retention_roi(
                audience_size=audience_size,
                campaign_cost_per_user=campaign_cost_per_user,
                retention_uplift_pct=uplift,
                contribution_per_retained_user=contribution_per_retained_user,
            ),
        }
        for uplift in retention_uplifts_pct
    ]
