"""Exploratory churn model restricted to eligible subscribers.

Features are measured through 2026-04-30.
The churn outcome covers cancellations from 2026-05-01 to 2026-06-30.

This is predictive analysis, not evidence of causal effects.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/ml_dataset.csv")
TARGET = "churn"

CANDIDATE_FEATURES = [
    "total_sessions",
    "active_days",
    "login_count",
    "product_view_count",
    "search_count",
    "add_to_cart_count",
    "days_since_last_activity",
    "days_since_last_login",
    "days_since_last_product_view",
    "events_per_active_day",
    "sessions_per_active_day",
    "search_rate",
    "cart_rate",
    "purchase_rate",
]


def prepare_subscriber_data(path=DATA_PATH):
    """Load data, retain subscribers, and select varying numeric features."""
    df = pd.read_csv(path)

    required = {TARGET, "subscription_count", *CANDIDATE_FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Churn must contain only non-missing 0/1 values.")

    subscribers = df.loc[df["subscription_count"] > 0].copy()

    if subscribers.empty:
        raise ValueError("No eligible subscribers found.")

    features = [
        col
        for col in CANDIDATE_FEATURES
        if pd.api.types.is_numeric_dtype(subscribers[col])
        and subscribers[col].nunique(dropna=True) > 1
    ]

    if not features:
        raise ValueError("No varying numeric features are available.")

    if subscribers[features].isna().any().any():
        raise ValueError("Selected features contain missing values.")

    if subscribers[TARGET].nunique() != 2:
        raise ValueError("Subscriber data must contain both churn classes.")

    return subscribers, features


def main():
    subscribers, features = prepare_subscriber_data()

    X = subscribers[features]
    y = subscribers[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    print("=" * 60)
    print("SUBSCRIBER-ONLY CHURN LOGISTIC REGRESSION")
    print("=" * 60)
    print(f"Eligible subscribers: {len(subscribers):,}")
    print(f"Observed churn events: {int(y.sum()):,}")
    print(f"Observed subscriber churn rate: {y.mean():.2%}")
    print(f"Training users: {len(X_train):,}")
    print(f"Test users: {len(X_test):,}")
    print(f"Test churn events: {int(y_test.sum()):,}")
    print(f"Features used ({len(features)}): {', '.join(features)}")

    print("\nHELD-OUT TEST METRICS")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(
        "Average precision: "
        f"{average_precision_score(y_test, probabilities):.3f}"
    )
    print(f"Test-set churn prevalence: {y_test.mean():.2%}")

    print("\nCONFUSION MATRIX (threshold = 0.50)")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, predictions, zero_division=0))

    coefficients = model.named_steps["logisticregression"].coef_[0]
    importance = pd.DataFrame(
        {
            "feature": features,
            "standardized_coefficient": coefficients,
            "odds_ratio_per_std_dev": pd.Series(coefficients).map(
                lambda value: float(__import__("math").exp(value))
            ),
        }
    )
    importance["abs_coefficient"] = importance["standardized_coefficient"].abs()
    importance = importance.sort_values("abs_coefficient", ascending=False)

    print("STANDARDIZED MODEL COEFFICIENTS")
    print(importance.drop(columns="abs_coefficient").round(3).to_string(index=False))

    print("\nINTERPRETATION NOTES")
    print("- The test set is small and contains relatively few churn events.")
    print("- Compare ROC-AUC with 0.50 and average precision with test churn prevalence.")
    print("- Coefficients describe associations, not causal effects.")
    print("- Class weighting affects probability calibration.")
    print("- Use only information available at the prediction date.")


if __name__ == "__main__":
    main()
