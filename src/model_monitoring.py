
"""Quality checks and monitoring summaries for churn predictions."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "user_id",
    "actual_churn",
    "churn_probability",
    "risk_segment",
}


def validate_predictions(df: pd.DataFrame) -> dict:
    """Validate prediction records against the model's risk thresholds."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    probability = pd.to_numeric(df["churn_probability"], errors="coerce")
    actual = pd.to_numeric(df["actual_churn"], errors="coerce")

    invalid_probability = (
        probability.isna()
        | ~np.isfinite(probability)
        | ~probability.between(0, 1)
    )
    invalid_actual = actual.isna() | ~actual.isin([0, 1])

    # Match pd.cut(bins=[-0.01, 0.30, 0.60, 1.00]) exactly.
    expected_risk = pd.cut(
        probability,
        bins=[-0.01, 0.30, 0.60, 1.00],
        labels=["Low", "Medium", "High"],
    ).astype("string")

    actual_risk = df["risk_segment"].astype("string")
    risk_mismatch = (
        actual_risk.ne(expected_risk)
        & ~invalid_probability
    )

    checks = {
        "row_count": int(len(df)),
        "missing_values": int(df[list(REQUIRED_COLUMNS)].isna().sum().sum()),
        "duplicate_user_ids": int(df["user_id"].duplicated().sum()),
        "invalid_probabilities": int(invalid_probability.sum()),
        "invalid_actual_labels": int(invalid_actual.sum()),
        "risk_segment_mismatches": int(risk_mismatch.sum()),
    }

    checks["passed"] = all(
        checks[key] == 0
        for key in (
            "missing_values",
            "duplicate_user_ids",
            "invalid_probabilities",
            "invalid_actual_labels",
            "risk_segment_mismatches",
        )
    )
    return checks


def summarize_predictions(df: pd.DataFrame) -> dict:
    """Summarize prediction scores, observed labels, and risk segments."""
    checks = validate_predictions(df)
    if not checks["passed"]:
        raise ValueError(f"Prediction validation failed: {checks}")

    probability = df["churn_probability"].astype(float)
    actual = df["actual_churn"].astype(int)

    return {
        "users": int(len(df)),
        "mean_predicted_probability": float(probability.mean()),
        "observed_churn_rate": float(actual.mean()),
        "prediction_observation_gap": float(
            probability.mean() - actual.mean()
        ),
        "risk_segment_counts": (
            df["risk_segment"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .astype(int)
            .to_dict()
        ),
    }
