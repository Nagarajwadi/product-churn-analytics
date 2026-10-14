"""
Hypothesis testing for a two-group churn experiment.

Input data must come from a properly designed experiment if the
results are to be interpreted causally. Statistical tests alone
cannot establish that an intervention caused an observed difference.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class ExperimentResult:
    control_users: int
    treatment_users: int
    control_churners: int
    treatment_churners: int
    control_rate: float
    treatment_rate: float
    absolute_difference: float
    relative_change: float
    z_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    statistically_significant: bool
    interpretation: str


def analyze_churn_experiment(
    control_users: int,
    control_churners: int,
    treatment_users: int,
    treatment_churners: int,
    alpha: float = 0.05,
) -> ExperimentResult:
    """Compare two independent churn proportions with a two-sided z-test."""
    if control_users <= 0 or treatment_users <= 0:
        raise ValueError("Each group must contain at least one user.")

    if not 0 <= control_churners <= control_users:
        raise ValueError("Control churners must be between 0 and control users.")

    if not 0 <= treatment_churners <= treatment_users:
        raise ValueError("Treatment churners must be between 0 and treatment users.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    p_control = control_churners / control_users
    p_treatment = treatment_churners / treatment_users
    difference = p_treatment - p_control

    pooled = (
        (control_churners + treatment_churners)
        / (control_users + treatment_users)
    )
    se_null = np.sqrt(
        pooled * (1 - pooled)
        * (1 / control_users + 1 / treatment_users)
    )

    if se_null == 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = difference / se_null
        p_value = float(2 * norm.sf(abs(z_stat)))

    # Unpooled normal-approximation confidence interval for the difference.
    se_difference = np.sqrt(
        p_control * (1 - p_control) / control_users
        + p_treatment * (1 - p_treatment) / treatment_users
    )
    z_critical = norm.ppf(1 - alpha / 2)
    ci_lower = difference - z_critical * se_difference
    ci_upper = difference + z_critical * se_difference

    significant = p_value < alpha
    relative_change = (
        difference / p_control if p_control > 0 else float("nan")
    )

    if significant:
        if difference < 0:
            interpretation = (
                "Treatment churn is statistically lower than control. "
                "Causal interpretation requires valid randomization and execution."
            )
        elif difference > 0:
            interpretation = (
                "Treatment churn is statistically higher than control. "
                "Causal interpretation requires valid randomization and execution."
            )
        else:
            interpretation = "The test reports a statistically significant difference."
    else:
        interpretation = (
            "Insufficient evidence to reject equal churn rates at this "
            "significance level. This does not prove the rates are equal."
        )

    return ExperimentResult(
        control_users=control_users,
        treatment_users=treatment_users,
        control_churners=control_churners,
        treatment_churners=treatment_churners,
        control_rate=p_control,
        treatment_rate=p_treatment,
        absolute_difference=difference,
        relative_change=relative_change,
        z_statistic=float(z_stat),
        p_value=p_value,
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=1 - alpha,
        statistically_significant=significant,
        interpretation=interpretation,
    )


def main():
    # Illustrative numbers only; not results from your project dataset.
    result = analyze_churn_experiment(
        control_users=1000,
        control_churners=100,
        treatment_users=1000,
        treatment_churners=80,
    )

    print("ILLUSTRATIVE EXPERIMENT ANALYSIS")
    print("--------------------------------")
    print(f"Control churn rate:   {result.control_rate:.2%}")
    print(f"Treatment churn rate: {result.treatment_rate:.2%}")
    print(f"Absolute difference:  {result.absolute_difference * 100:.2f} percentage points")
    print(f"Relative change:      {result.relative_change:.2%}")
    print(f"Z statistic:          {result.z_statistic:.3f}")
    print(f"P-value:              {result.p_value:.4f}")
    print(
        f"{result.confidence_level:.0%} CI for difference: "
        f"[{result.ci_lower * 100:.2f}, {result.ci_upper * 100:.2f}] "
        "percentage points"
    )
    print(f"Statistically significant: {result.statistically_significant}")
    print(f"Interpretation: {result.interpretation}")
    print("\nIllustrative values only; not evidence of real product impact.")


if __name__ == "__main__":
    main()
