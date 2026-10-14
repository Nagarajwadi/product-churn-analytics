"""Reproducible randomized A/B experiment simulation.

Uses subscriber records for baseline characteristics but generates new,
synthetic outcomes. Results do not represent real product impact.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


DATA_PATH = Path("data/processed/ml_dataset.csv")
SEED = 42
CONTROL_CHURN_RATE = 0.085
TREATMENT_CHURN_RATE = 0.065

BASELINE_FEATURES = [
    "total_sessions",
    "active_days",
    "days_since_last_activity",
    "days_since_last_login",
    "days_since_last_product_view",
    "search_rate",
    "cart_rate",
]


def load_subscribers(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {"user_id", "subscription_count", *BASELINE_FEATURES}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    subscribers = df.loc[df["subscription_count"] > 0].copy()

    if subscribers.empty:
        raise ValueError("No eligible subscribers found.")
    if subscribers["user_id"].duplicated().any():
        raise ValueError("Each subscriber must appear only once.")
    if subscribers[BASELINE_FEATURES].isna().any().any():
        raise ValueError("Baseline features contain missing values.")

    return subscribers.reset_index(drop=True)


def standardized_mean_difference(control, treatment):
    """Calculate standardized mean differences for numeric baseline features."""
    rows = []

    for feature in BASELINE_FEATURES:
        control_values = control[feature].astype(float)
        treatment_values = treatment[feature].astype(float)

        control_var = control_values.var(ddof=1)
        treatment_var = treatment_values.var(ddof=1)
        pooled_sd = np.sqrt((control_var + treatment_var) / 2)

        smd = (
            (treatment_values.mean() - control_values.mean()) / pooled_sd
            if pooled_sd > 0
            else 0.0
        )

        rows.append({
            "feature": feature,
            "control_mean": control_values.mean(),
            "treatment_mean": treatment_values.mean(),
            "smd": smd,
            "abs_smd": abs(smd),
        })

    return pd.DataFrame(rows).sort_values("abs_smd", ascending=False)


def compare_binary_outcomes(control, treatment):
    """Estimate treatment-minus-control risk difference, CI, and two-sided z-test."""
    n_control = len(control)
    n_treatment = len(treatment)
    events_control = int(control.sum())
    events_treatment = int(treatment.sum())

    rate_control = events_control / n_control
    rate_treatment = events_treatment / n_treatment
    difference = rate_treatment - rate_control

    # Unpooled Wald confidence interval for the risk difference.
    se_difference = np.sqrt(
        rate_control * (1 - rate_control) / n_control
        + rate_treatment * (1 - rate_treatment) / n_treatment
    )
    ci_low = difference - 1.96 * se_difference
    ci_high = difference + 1.96 * se_difference

    # Pooled two-proportion z-test under equal outcome rates.
    pooled_rate = (events_control + events_treatment) / (
        n_control + n_treatment
    )
    se_null = np.sqrt(
        pooled_rate * (1 - pooled_rate)
        * (1 / n_control + 1 / n_treatment)
    )

    if se_null == 0:
        z_statistic = 0.0
        p_value = 1.0
    else:
        z_statistic = difference / se_null
        p_value = float(2 * norm.sf(abs(z_statistic)))

    return {
        "control_users": n_control,
        "treatment_users": n_treatment,
        "control_events": events_control,
        "treatment_events": events_treatment,
        "control_rate": rate_control,
        "treatment_rate": rate_treatment,
        "risk_difference": difference,
        "relative_change": (
            difference / rate_control if rate_control > 0 else np.nan
        ),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "z_statistic": z_statistic,
        "p_value": p_value,
    }


def run_experiment(path=DATA_PATH, seed=SEED):
    subscribers = load_subscribers(path)
    rng = np.random.default_rng(seed)
    n = len(subscribers)

    # Assign approximately half the subscribers to each arm at random.
    assignment = np.zeros(n, dtype=int)
    assignment[rng.permutation(n)[: n // 2]] = 1

    experiment = subscribers.copy()
    experiment["treatment"] = assignment

    control = experiment.loc[experiment["treatment"] == 0].copy()
    treatment = experiment.loc[experiment["treatment"] == 1].copy()

    # Simulate a new outcome; do not reuse the historical churn label.
    probabilities = np.where(
        experiment["treatment"].to_numpy() == 1,
        TREATMENT_CHURN_RATE,
        CONTROL_CHURN_RATE,
    )
    experiment["simulated_churn"] = rng.binomial(1, probabilities)

    control_outcomes = experiment.loc[
        experiment["treatment"] == 0, "simulated_churn"
    ]
    treatment_outcomes = experiment.loc[
        experiment["treatment"] == 1, "simulated_churn"
    ]

    balance = standardized_mean_difference(control, treatment)
    result = compare_binary_outcomes(control_outcomes, treatment_outcomes)

    return experiment, balance, result


def main():
    experiment, balance, result = run_experiment()

    print("=" * 65)
    print("SIMULATED RANDOMIZED A/B EXPERIMENT")
    print("=" * 65)
    print("Outcome: synthetic cancellation indicator")
    print("Historical churn labels are not used as experiment outcomes.")
    print(f"Random seed: {SEED}")
    print(f"Subscribers randomized: {len(experiment):,}")
    print(f"Control users: {result['control_users']:,}")
    print(f"Treatment users: {result['treatment_users']:,}")
    print(
        "Simulated rates by design: "
        f"control={CONTROL_CHURN_RATE:.1%}, "
        f"treatment={TREATMENT_CHURN_RATE:.1%}"
    )

    print("\nBASELINE BALANCE")
    print(balance.drop(columns="abs_smd").round(3).to_string(index=False))
    print(f"\nLargest absolute SMD: {balance['abs_smd'].max():.3f}")
    print("Rule of thumb: absolute SMD below 0.10 is often considered small.")
    print("Balance is a diagnostic, not a test of whether randomization occurred.")

    print("\nSIMULATED OUTCOME RESULTS")
    print(
        f"Control:   {result['control_events']}/{result['control_users']} "
        f"({result['control_rate']:.2%})"
    )
    print(
        f"Treatment: {result['treatment_events']}/{result['treatment_users']} "
        f"({result['treatment_rate']:.2%})"
    )
    print(
        "Absolute risk difference (treatment - control): "
        f"{result['risk_difference'] * 100:.2f} percentage points"
    )
    print(f"95% CI: [{result['ci_low'] * 100:.2f}, "
          f"{result['ci_high'] * 100:.2f}] percentage points")
    print(f"Relative change: {result['relative_change']:.1%}")
    print(f"Two-sided z-test p-value: {result['p_value']:.4f}")

    if result["p_value"] < 0.05:
        print("Decision: reject equal outcome rates at alpha = 0.05.")
    else:
        print("Decision: insufficient evidence to reject equal outcome rates.")

    print("\nLIMITATIONS")
    print("- Outcomes are simulated, not observed product behavior.")
    print("- The treatment effect is imposed by the simulation design.")
    print("- One run does not establish statistical power or real-world impact.")
    print("- The confidence interval uses a normal approximation.")


if __name__ == "__main__":
    main()
