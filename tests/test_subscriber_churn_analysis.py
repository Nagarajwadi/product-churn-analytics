import pandas as pd
import pytest

from src.subscriber_churn_analysis import prepare_subscriber_data


def make_dataset():
    rows = []
    for i in range(30):
        rows.append({
            "user_id": f"U{i:03d}",
            "subscription_count": 1 if i < 20 else 0,
            "churn": 1 if i in (0, 1, 2, 3) else 0,
            "total_sessions": i + 1,
            "active_days": (i % 5) + 1,
            "login_count": i + 2,
            "product_view_count": i + 3,
            "search_count": i % 4,
            "add_to_cart_count": i % 3,
            "days_since_last_activity": i + 1,
            "days_since_last_login": i + 2,
            "days_since_last_product_view": i + 3,
            "events_per_active_day": i / 10 + 1,
            "sessions_per_active_day": i / 20 + 1,
            "search_rate": i / 100,
            "cart_rate": i / 200,
            "purchase_rate": 0.05,
            "purchase_count": 1,
        })
    return pd.DataFrame(rows)


def test_keeps_only_subscribers_and_excludes_constant_features(tmp_path):
    path = tmp_path / "dataset.csv"
    make_dataset().to_csv(path, index=False)

    subscribers, features = prepare_subscriber_data(path)

    assert len(subscribers) == 20
    assert (subscribers["subscription_count"] > 0).all()
    assert "subscription_count" not in features
    assert "purchase_count" not in features
    assert "total_sessions" in features


def test_rejects_missing_required_columns(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"churn": [0, 1]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        prepare_subscriber_data(path)


def test_rejects_missing_feature_values(tmp_path):
    df = make_dataset()
    df.loc[0, "total_sessions"] = None
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing values"):
        prepare_subscriber_data(path)


def test_rejects_dataset_without_subscribers(tmp_path):
    df = make_dataset()
    df["subscription_count"] = 0
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="No eligible subscribers"):
        prepare_subscriber_data(path)
