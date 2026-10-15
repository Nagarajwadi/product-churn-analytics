
import pandas as pd
import pytest

from src.model_monitoring import (
    summarize_predictions,
    validate_predictions,
)


def sample_predictions():
    return pd.DataFrame({
        "user_id": ["U1", "U2", "U3", "U4"],
        "actual_churn": [0, 1, 0, 0],
        "churn_probability": [0.10, 0.80, 0.30, 0.50],
        "risk_segment": ["Low", "High", "Low", "Medium"],
    })


def test_valid_predictions_pass():
    result = validate_predictions(sample_predictions())
    assert result["passed"] is True
    assert result["row_count"] == 4
    assert result["duplicate_user_ids"] == 0


def test_duplicate_user_ids_are_detected():
    df = sample_predictions()
    df.loc[1, "user_id"] = "U1"

    result = validate_predictions(df)
    assert result["duplicate_user_ids"] == 1
    assert result["passed"] is False


def test_invalid_probability_is_detected():
    df = sample_predictions()
    df.loc[0, "churn_probability"] = 1.5

    result = validate_predictions(df)
    assert result["invalid_probabilities"] == 1
    assert result["passed"] is False


def test_risk_segment_mismatch_is_detected():
    df = sample_predictions()
    df.loc[0, "risk_segment"] = "High"

    result = validate_predictions(df)
    assert result["risk_segment_mismatches"] == 1
    assert result["passed"] is False


def test_summary_reports_rates_and_segment_counts():
    result = summarize_predictions(sample_predictions())

    assert result["users"] == 4
    assert result["observed_churn_rate"] == pytest.approx(0.25)
    assert result["mean_predicted_probability"] == pytest.approx(0.425)
    assert result["risk_segment_counts"] == {
        "Low": 2,
        "Medium": 1,
        "High": 1,
    }


def test_summary_rejects_invalid_predictions():
    df = sample_predictions()
    df.loc[0, "actual_churn"] = 2

    with pytest.raises(ValueError, match="Prediction validation failed"):
        summarize_predictions(df)
