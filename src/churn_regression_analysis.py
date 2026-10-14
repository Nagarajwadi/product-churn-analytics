"""
Exploratory logistic regression for historical customer churn.

This analysis estimates associations in observational data.
It does not estimate the causal impact of changing a product feature.
"""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ml_dataset.csv"

FEATURES = [
    "days_since_last_activity",
    "total_sessions",
    "subscription_count",
]
TARGET = "churn"


def main():
    df = pd.read_csv(DATA_PATH)

    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Selected features or target contain missing values.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Churn must contain only 0/1 values.")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Scaling is fitted only on training data to avoid test-set leakage.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilities)
    average_precision = average_precision_score(y_test, probabilities)

    coefficients = model.named_steps["logisticregression"].coef_[0]
    coefficient_table = pd.DataFrame(
        {
            "feature": FEATURES,
            "standardized_coefficient": coefficients,
            "odds_ratio_per_standard_deviation": [
                float(__import__("math").exp(value))
                for value in coefficients
            ],
        }
    ).sort_values(
        "standardized_coefficient",
        ascending=False,
    )

    print("EXPLORATORY CHURN LOGISTIC REGRESSION")
    print("-------------------------------------")
    print(f"Dataset users: {len(df):,}")
    print(f"Churn events: {int(y.sum()):,}")
    print(f"Training users: {len(X_train):,}")
    print(f"Test users: {len(X_test):,}")
    print(f"Test churn events: {int(y_test.sum()):,}")
    print(f"Test ROC-AUC: {roc_auc:.3f}")
    print(f"Test average precision: {average_precision:.3f}")
    print("\nAdjusted model associations:")
    print(coefficient_table.to_string(index=False, formatters={
        "standardized_coefficient": "{:.3f}".format,
        "odds_ratio_per_standard_deviation": "{:.3f}".format,
    }))
    print("\nInterpretation cautions:")
    print("- Odds ratios describe model associations, not causal effects.")
    print("- Class weighting affects coefficient interpretation.")
    print("- Results may be unstable with only 90 churn events.")
    print("- Evaluate against the churn prevalence and a baseline model.")


if __name__ == "__main__":
    main()
