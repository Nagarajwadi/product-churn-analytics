import pandas as pd


DATA_FILE = "data/processed/ml_dataset.csv"

EXPECTED_FEATURES = [
    "total_events",
    "total_sessions",
    "active_days",
    "login_count",
    "product_view_count",
    "search_count",
    "add_to_cart_count",
    "purchase_count",
    "subscription_count",
    "days_since_last_activity",
    "days_since_last_login",
    "days_since_last_product_view",
    "events_per_active_day",
    "sessions_per_active_day",
    "search_rate",
    "cart_rate",
    "purchase_rate",
]


def test_ml_dataset_has_expected_columns():
    df = pd.read_csv(DATA_FILE)

    expected_columns = ["user_id", *EXPECTED_FEATURES, "churn"]

    assert list(df.columns) == expected_columns


def test_ml_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_FILE)

    assert df.isna().sum().sum() == 0


def test_churn_target_is_binary():
    df = pd.read_csv(DATA_FILE)

    assert set(df["churn"].unique()).issubset({0, 1})
