import pytest

from src.hypothesis_testing import analyze_churn_experiment


def test_equal_rates_are_not_significant():
    result = analyze_churn_experiment(
        control_users=1000,
        control_churners=100,
        treatment_users=1000,
        treatment_churners=100,
    )

    assert result.control_rate == result.treatment_rate
    assert result.p_value == pytest.approx(1.0)
    assert result.statistically_significant is False


def test_large_difference_is_detected():
    result = analyze_churn_experiment(
        control_users=5000,
        control_churners=500,
        treatment_users=5000,
        treatment_churners=350,
    )

    assert result.absolute_difference == pytest.approx(-0.03)
    assert result.p_value < 0.05
    assert result.statistically_significant is True
    assert result.ci_upper < 0


def test_invalid_group_size_is_rejected():
    with pytest.raises(ValueError):
        analyze_churn_experiment(
            control_users=0,
            control_churners=0,
            treatment_users=100,
            treatment_churners=10,
        )


def test_churners_cannot_exceed_group_size():
    with pytest.raises(ValueError):
        analyze_churn_experiment(
            control_users=100,
            control_churners=101,
            treatment_users=100,
            treatment_churners=10,
        )


def test_invalid_significance_level_is_rejected():
    with pytest.raises(ValueError):
        analyze_churn_experiment(
            control_users=100,
            control_churners=10,
            treatment_users=100,
            treatment_churners=8,
            alpha=1.0,
        )
