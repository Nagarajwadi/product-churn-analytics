import pytest

from src.retention_roi import (
    calculate_retention_roi,
    compare_retention_scenarios,
)


def test_calculate_retention_roi():
    result = calculate_retention_roi(
        audience_size=100,
        campaign_cost_per_user=20,
        retention_uplift_pct=5,
        contribution_per_retained_user=500,
    )

    assert result["campaign_cost"] == 2000
    assert result["incremental_retained_users"] == 5
    assert result["incremental_contribution"] == 2500
    assert result["net_benefit"] == 500
    assert result["roi_pct"] == 25
    assert result["break_even_uplift_pct"] == 4


def test_zero_campaign_cost_has_no_roi_percentage():
    result = calculate_retention_roi(100, 0, 5, 500)

    assert result["campaign_cost"] == 0
    assert result["roi_pct"] is None


def test_zero_contribution_has_no_break_even_uplift():
    result = calculate_retention_roi(100, 20, 5, 0)

    assert result["break_even_uplift_pct"] is None
    assert result["net_benefit"] == -2000


def test_zero_audience_is_supported():
    result = calculate_retention_roi(0, 20, 5, 500)

    assert result["campaign_cost"] == 0
    assert result["incremental_retained_users"] == 0
    assert result["roi_pct"] is None


@pytest.mark.parametrize("audience_size", [-1, 1.5, True])
def test_invalid_audience_size_is_rejected(audience_size):
    with pytest.raises(ValueError):
        calculate_retention_roi(audience_size, 20, 5, 500)


@pytest.mark.parametrize(
    "uplift",
    [-1, 100.1],
)
def test_invalid_retention_uplift_is_rejected(uplift):
    with pytest.raises(ValueError):
        calculate_retention_roi(100, 20, uplift, 500)


def test_negative_campaign_cost_is_rejected():
    with pytest.raises(ValueError):
        calculate_retention_roi(100, -1, 5, 500)


def test_negative_contribution_is_rejected():
    with pytest.raises(ValueError):
        calculate_retention_roi(100, 20, 5, -1)


def test_sensitivity_scenarios():
    results = compare_retention_scenarios(
        audience_size=100,
        campaign_cost_per_user=20,
        retention_uplifts_pct=[2, 5, 8],
        contribution_per_retained_user=500,
    )

    assert [row["net_benefit"] for row in results] == [
        -1000,
        500,
        2000,
    ]
    assert [row["roi_pct"] for row in results] == [
        -50,
        25,
        100,
    ]


def test_empty_scenarios_are_rejected():
    with pytest.raises(ValueError):
        compare_retention_scenarios(100, 20, [], 500)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_cost_per_user", float("nan")),
        ("campaign_cost_per_user", float("inf")),
        ("retention_uplift_pct", float("nan")),
        ("retention_uplift_pct", float("inf")),
        ("contribution_per_retained_user", float("nan")),
        ("contribution_per_retained_user", float("inf")),
    ],
)
def test_non_finite_financial_inputs_are_rejected(field, value):
    inputs = {
        "audience_size": 100,
        "campaign_cost_per_user": 20,
        "retention_uplift_pct": 5,
        "contribution_per_retained_user": 500,
    }
    inputs[field] = value

    with pytest.raises(ValueError):
        calculate_retention_roi(**inputs)
