"""
Simulated A/B test analysis for the product churn analytics project.

IMPORTANT:
This is a statistical demonstration using historical churn labels.
There is no real experiment assignment or intervention in the dataset.
Results must not be interpreted as evidence that a treatment reduces churn.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ml_dataset.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95


def wilson_interval(successes: int, total: int, z: float) -> tuple[float, float]:
    """Calculate a Wilson confidence interval for a proportion."""
    if total == 0:
        return np.nan, np.nan

    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(
            proportion * (1 - proportion) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )
    return center - margin, center + margin


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required_columns = {"user_id", "churn"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df["user_id"].duplicated().any():
        raise ValueError("Expected one row per user; duplicate user IDs found.")

    if df["churn"].isna().any() or not set(df["churn"].unique()).issubset({0, 1}):
        raise ValueError("The churn column must contain only non-missing 0/1 values.")

    # Reproducible 50/50 assignment for demonstration only.
    rng = np.random.default_rng(RANDOM_SEED)
    df = df.copy()
    df["experiment_group"] = rng.choice(
        ["control", "treatment"],
        size=len(df),
        replace=True,
    )

    summary = (
        df.groupby("experiment_group", observed=True)
        .agg(
            users=("user_id", "nunique"),
            churners=("churn", "sum"),
            churn_rate=("churn", "mean"),
        )
        .reindex(["control", "treatment"])
        .reset_index()
    )

    control = summary.loc[summary["experiment_group"] == "control"].iloc[0]
    treatment = summary.loc[summary["experiment_group"] == "treatment"].iloc[0]

    n_control = int(control["users"])
    n_treatment = int(treatment["users"])
    x_control = int(control["churners"])
    x_treatment = int(treatment["churners"])
    p_control = float(control["churn_rate"])
    p_treatment = float(treatment["churn_rate"])

    # Difference is treatment minus control. Negative means lower observed
    # churn in the simulated treatment group, not a causal treatment effect.
    absolute_difference = p_treatment - p_control
    relative_difference = (
        absolute_difference / p_control if p_control > 0 else np.nan
    )

    # Two-proportion z-test under the null hypothesis of equal churn rates.
    pooled_rate = (x_control + x_treatment) / (n_control + n_treatment)
    standard_error_null = np.sqrt(
        pooled_rate
        * (1 - pooled_rate)
        * (1 / n_control + 1 / n_treatment)
    )

    if standard_error_null > 0:
        z_statistic = absolute_difference / standard_error_null
        p_value = float(2 * norm.sf(abs(z_statistic)))
    else:
        z_statistic = 0.0
        p_value = 1.0

    # Unpooled Wald interval for the absolute difference.
    standard_error_difference = np.sqrt(
        p_control * (1 - p_control) / n_control
        + p_treatment * (1 - p_treatment) / n_treatment
    )
    z_critical = norm.ppf(1 - (1 - CONFIDENCE_LEVEL) / 2)
    ci_lower = absolute_difference - z_critical * standard_error_difference
    ci_upper = absolute_difference + z_critical * standard_error_difference

    summary["churn_rate_pct"] = summary["churn_rate"] * 100
    summary["confidence_interval_lower_pct"] = [
        wilson_interval(x_control, n_control, z_critical)[0] * 100,
        wilson_interval(x_treatment, n_treatment, z_critical)[0] * 100,
    ]
    summary["confidence_interval_upper_pct"] = [
        wilson_interval(x_control, n_control, z_critical)[1] * 100,
        wilson_interval(x_treatment, n_treatment, z_critical)[1] * 100,
    ]

    results = pd.DataFrame(
        [
            {
                "analysis_type": "SIMULATED DEMONSTRATION — NOT A REAL EXPERIMENT",
                "control_users": n_control,
                "treatment_users": n_treatment,
                "control_churners": x_control,
                "treatment_churners": x_treatment,
                "control_churn_rate_pct": p_control * 100,
                "treatment_churn_rate_pct": p_treatment * 100,
                "absolute_difference_percentage_points": absolute_difference * 100,
                "relative_difference_pct": relative_difference * 100,
                "z_statistic": z_statistic,
                "p_value": p_value,
                "difference_ci_lower_percentage_points": ci_lower * 100,
                "difference_ci_upper_percentage_points": ci_upper * 100,
                "confidence_level": CONFIDENCE_LEVEL,
                "random_seed": RANDOM_SEED,
                "causal_interpretation": "Not supported: no real intervention or experiment assignment",
            }
        ]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "ab_test_group_summary.csv"
    results_path = OUTPUT_DIR / "ab_test_results.csv"

    summary.to_csv(summary_path, index=False)
    results.to_csv(results_path, index=False)

    print("\nSIMULATED A/B TEST — DEMONSTRATION ONLY")
    print("Historical churn labels were randomly split into two groups.")
    print("This does not measure the effect of a real intervention.\n")
    print(
        summary[
            ["experiment_group", "users", "churners", "churn_rate_pct"]
        ].to_string(index=False, formatters={"churn_rate_pct": "{:.3f}%".format})
    )
    print(f"\nAbsolute difference (treatment - control): {absolute_difference * 100:.3f} percentage points")
    print(f"Relative difference: {relative_difference * 100:.2f}%")
    print(f"Two-proportion z-test p-value: {p_value:.4f}")
    print(
        f"{CONFIDENCE_LEVEL:.0%} CI for difference: "
        f"[{ci_lower * 100:.3f}, {ci_upper * 100:.3f}] percentage points"
    )
    print("\nInterpretation: these are simulated-group comparisons, not causal evidence.")
    print(f"\nSaved group summary: {summary_path}")
    print(f"Saved test results: {results_path}")


if __name__ == "__main__":
    main()