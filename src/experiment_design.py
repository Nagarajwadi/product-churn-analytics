"""
A/B test sample-size planning for two independent proportions.

Educational example: replace illustrative assumptions with a measured
baseline and a business-approved minimum detectable effect (MDE).
This planner does not analyze or prove the impact of a real experiment.
"""

import math

from scipy.stats import norm


def _validate_probability(name: str, value: float) -> None:
    """Validate a finite probability strictly between zero and one."""
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and between 0 and 1.")


def calculate_sample_size(
    baseline_rate: float,
    treatment_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Estimate usable users required per group for a two-sided test."""
    _validate_probability("baseline_rate", baseline_rate)
    _validate_probability("treatment_rate", treatment_rate)
    _validate_probability("alpha", alpha)
    _validate_probability("power", power)

    if treatment_rate == baseline_rate:
        raise ValueError("Treatment rate must differ from baseline rate.")

    pooled_rate = (baseline_rate + treatment_rate) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)

    numerator = (
        z_alpha * math.sqrt(2 * pooled_rate * (1 - pooled_rate))
        + z_power
        * math.sqrt(
            baseline_rate * (1 - baseline_rate)
            + treatment_rate * (1 - treatment_rate)
        )
    ) ** 2

    denominator = (treatment_rate - baseline_rate) ** 2
    return math.ceil(numerator / denominator)


def calculate_recruitment_target(
    users_per_group: int,
    attrition_rate: float = 0.0,
) -> int:
    """Estimate recruits per group needed after expected attrition."""
    if isinstance(users_per_group, bool) or not isinstance(users_per_group, int):
        raise ValueError("users_per_group must be a positive integer.")
    if users_per_group <= 0:
        raise ValueError("users_per_group must be a positive integer.")
    if not math.isfinite(attrition_rate) or not 0 <= attrition_rate < 1:
        raise ValueError("attrition_rate must be finite and between 0 and 1.")

    return math.ceil(users_per_group / (1 - attrition_rate))


def compare_sample_size_scenarios(
    baseline_rate: float,
    treatment_rates: list[float],
    alpha: float = 0.05,
    power: float = 0.80,
    attrition_rate: float = 0.0,
) -> list[dict]:
    """Compare sample-size and recruitment targets across treatment scenarios."""
    if not treatment_rates:
        raise ValueError("treatment_rates must contain at least one rate.")

    scenarios = []

    for treatment_rate in treatment_rates:
        users_per_group = calculate_sample_size(
            baseline_rate=baseline_rate,
            treatment_rate=treatment_rate,
            alpha=alpha,
            power=power,
        )
        recruits_per_group = calculate_recruitment_target(
            users_per_group=users_per_group,
            attrition_rate=attrition_rate,
        )

        scenarios.append({
            "baseline_rate": baseline_rate,
            "treatment_rate": treatment_rate,
            "absolute_effect_pp": (treatment_rate - baseline_rate) * 100,
            "users_per_group": users_per_group,
            "recruits_per_group": recruits_per_group,
            "total_recruits": 2 * recruits_per_group,
        })

    return scenarios


def main():
    # Illustrative assumptions, not measured project results.
    baseline = 0.10
    target = 0.08
    alpha = 0.05
    power = 0.80
    expected_attrition = 0.10

    users_per_group = calculate_sample_size(
        baseline_rate=baseline,
        treatment_rate=target,
        alpha=alpha,
        power=power,
    )

    recruits_per_group = calculate_recruitment_target(
        users_per_group,
        expected_attrition,
    )

    print("A/B TEST SAMPLE-SIZE PLAN")
    print("-------------------------")
    print(f"Illustrative baseline rate: {baseline:.1%}")
    print(f"Target treatment rate:      {target:.1%}")
    print(f"Absolute improvement:        {(baseline - target) * 100:.1f} percentage points")
    print(f"Relative reduction:          {(baseline - target) / baseline:.1%}")
    print(f"Significance level (alpha):  {alpha:.0%}")
    print(f"Statistical power:           {power:.0%}")
    print(f"Usable users per group:      {users_per_group:,}")
    print(f"Expected attrition:          {expected_attrition:.0%}")
    print(f"Recruit control users:       {recruits_per_group:,}")
    print(f"Recruit treatment users:     {recruits_per_group:,}")
    print(f"Total recruitment target:    {2 * recruits_per_group:,}")
    print()

    scenarios = compare_sample_size_scenarios(
        baseline_rate=baseline,
        treatment_rates=[0.095, 0.09, 0.08, 0.07],
        alpha=alpha,
        power=power,
        attrition_rate=expected_attrition,
    )

    print("\nSCENARIO COMPARISON")
    print("-" * 75)
    print(
        f"{'Target':>8} {'Reduction':>12} "
        f"{'Usable/group':>15} {'Recruit/group':>15} {'Total':>10}"
    )

    for scenario in scenarios:
        reduction_pp = (
            scenario["baseline_rate"] - scenario["treatment_rate"]
        ) * 100

        print(
            f"{scenario['treatment_rate']:>7.1%} "
            f"{reduction_pp:>11.2f} pp "
            f"{scenario['users_per_group']:>15,} "
            f"{scenario['recruits_per_group']:>15,} "
            f"{scenario['total_recruits']:>10,}"
        )


    print("Assumptions: independent groups, equal allocation, two-sided")
    print("test, and normal-approximation sample-size calculation.")
    print("Attrition adjustment is an estimate; actual retention may differ.")
    print("Planning estimates are not evidence of real product impact.")


if __name__ == "__main__":
    main()
