"""
A/B test sample-size planning for two independent proportions.

Educational example: replace the illustrative assumptions with a
measured baseline and a business-approved minimum detectable effect.
This planner does not analyze or prove the impact of a real experiment.
"""

import math
from scipy.stats import norm


def calculate_sample_size(
    baseline_rate: float,
    treatment_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Approximate required users per group for a two-sided test."""
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be between 0 and 1.")
    if not 0 < treatment_rate < 1:
        raise ValueError("treatment_rate must be between 0 and 1.")
    if treatment_rate == baseline_rate:
        raise ValueError("Treatment rate must differ from baseline rate.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1.")

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


def main():
    # Illustrative assumptions, not measured project results.
    baseline = 0.10
    target = 0.08
    alpha = 0.05
    power = 0.80

    users_per_group = calculate_sample_size(
        baseline_rate=baseline,
        treatment_rate=target,
        alpha=alpha,
        power=power,
    )

    print("A/B TEST SAMPLE-SIZE PLAN")
    print("-------------------------")
    print(f"Illustrative baseline rate: {baseline:.1%}")
    print(f"Target treatment rate:      {target:.1%}")
    print(f"Absolute improvement:        {(baseline - target) * 100:.1f} percentage points")
    print(f"Relative reduction:          {(baseline - target) / baseline:.1%}")
    print(f"Significance level (alpha):  {alpha:.0%}")
    print(f"Statistical power:           {power:.0%}")
    print(f"Required control users:      {users_per_group:,}")
    print(f"Required treatment users:    {users_per_group:,}")
    print(f"Total required users:        {2 * users_per_group:,}")
    print()
    print("Assumes independent groups, equal allocation, a two-sided")
    print("test, and a normal-approximation calculation.")
    print("Actual experiments may need more users for attrition,")
    print("multiple comparisons, or other design considerations.")


if __name__ == "__main__":
    main()
