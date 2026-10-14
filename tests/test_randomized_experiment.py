import numpy as np
import pandas as pd
import pytest

from src.randomized_experiment import (
    BASELINE_FEATURES,
    compare_binary_outcomes,
    load_subscribers,
    run_experiment,
    standardized_mean_difference,
)


def make_dataset():
    rows = []
    for i in range(40):
        row = {
            "user_id": f"U{i:03d}",
            "subscription_count": 1 if i < 30 else 0,
        }
        for j, feature in enumerate(BASELINE_FEATURES):
            row[feature] = float(i + j)
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_subscribers_filters_non_subscribers(tmp_path):
    path = tmp_path / "dataset.csv"
    make_dataset().to_csv(path, index=False)

    subscribers = load_subscribers(path)

    assert len(subscribers) == 30
    assert (subscribers["subscription_count"] > 0).all()
    assert subscribers["user_id"].is_unique


def test_randomization_is_reproducible_and_balanced_in_size(tmp_path):
    path = tmp_path / "dataset.csv"
    make_dataset().to_csv(path, index=False)

    first, _, _ = run_experiment(path, seed=42)
    second, _, _ = run_experiment(path, seed=42)

    assert first["treatment"].tolist() == second["treatment"].tolist()
    assert first["simulated_churn"].tolist() == second["simulated_churn"].tolist()
    assert first["treatment"].sum() == 15
    assert len(first) - first["treatment"].sum() == 15


def test_smd_is_zero_for_identical_groups():
    values = pd.DataFrame({
        feature: np.arange(10, dtype=float)
        for feature in BASELINE_FEATURES
    })

    balance = standardized_mean_difference(values, values.copy())

    assert (balance["smd"].abs() < 1e-12).all()


def test_binary_outcome_comparison_returns_valid_p_value():
    control = pd.Series([0] * 90 + [1] * 10)
    treatment = pd.Series([0] * 92 + [1] * 8)

    result = compare_binary_outcomes(control, treatment)

    assert result["control_users"] == 100
    assert result["treatment_users"] == 100
    assert result["risk_difference"] == pytest.approx(-0.02)
    assert 0 <= result["p_value"] <= 1
    assert result["ci_low"] < result["ci_high"]


def test_load_subscribers_rejects_duplicate_users(tmp_path):
    df = make_dataset()
    df.loc[1, "user_id"] = df.loc[0, "user_id"]
    path = tmp_path / "dataset.csv"
    df.to_csv(path, index=False)

    with pytest.raises(ValueError, match="Each subscriber must appear only once"):
        load_subscribers(path)
