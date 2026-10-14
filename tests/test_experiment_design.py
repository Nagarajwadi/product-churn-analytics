import pytest

from src.experiment_design import (
    calculate_recruitment_target,
    calculate_sample_size,
    compare_sample_size_scenarios,
)


def test_sample_size_matches_expected_example():
    result = calculate_sample_size(0.10, 0.08, alpha=0.05, power=0.80)
    assert result == 3213


def test_sample_size_increases_for_smaller_effect():
    larger_effect = calculate_sample_size(0.10, 0.08)
    smaller_effect = calculate_sample_size(0.10, 0.09)
    assert smaller_effect > larger_effect


@pytest.mark.parametrize(
    ("baseline", "treatment", "alpha", "power"),
    [
        (0.0, 0.08, 0.05, 0.80),
        (0.10, 1.0, 0.05, 0.80),
        (0.10, 0.08, 1.0, 0.80),
        (0.10, 0.08, 0.05, 0.0),
        (float("nan"), 0.08, 0.05, 0.80),
        (0.10, float("inf"), 0.05, 0.80),
        (0.10, 0.08, float("nan"), 0.80),
        (0.10, 0.08, 0.05, float("inf")),
    ],
)
def test_sample_size_rejects_invalid_probabilities(
    baseline, treatment, alpha, power
):
    with pytest.raises(ValueError):
        calculate_sample_size(baseline, treatment, alpha, power)


def test_sample_size_rejects_equal_rates():
    with pytest.raises(ValueError):
        calculate_sample_size(0.10, 0.10)


def test_recruitment_target_without_attrition():
    assert calculate_recruitment_target(3213) == 3213


def test_recruitment_target_adjusts_for_attrition():
    assert calculate_recruitment_target(3213, 0.10) == 3570


@pytest.mark.parametrize("attrition", [-0.01, 1.0, float("nan"), float("inf")])
def test_recruitment_target_rejects_invalid_attrition(attrition):
    with pytest.raises(ValueError):
        calculate_recruitment_target(100, attrition)


@pytest.mark.parametrize("sample_size", [0, -1, 1.5, True])
def test_recruitment_target_rejects_invalid_sample_size(sample_size):
    with pytest.raises(ValueError):
        calculate_recruitment_target(sample_size)


def test_recruitment_target_rejects_non_integer_sample_size():
    with pytest.raises(ValueError):
        calculate_recruitment_target(100.0)


def test_scenario_comparison_orders_sample_sizes_by_effect():
    scenarios = compare_sample_size_scenarios(
        baseline_rate=0.10,
        treatment_rates=[0.095, 0.09, 0.08, 0.07],
    )

    sizes = [scenario["users_per_group"] for scenario in scenarios]

    assert sizes[0] > sizes[1] > sizes[2] > sizes[3]


def test_scenario_comparison_includes_attrition_adjustment():
    scenarios = compare_sample_size_scenarios(
        baseline_rate=0.10,
        treatment_rates=[0.08],
        attrition_rate=0.10,
    )

    assert scenarios[0]["users_per_group"] == 3213
    assert scenarios[0]["recruits_per_group"] == 3570
    assert scenarios[0]["total_recruits"] == 7140


def test_scenario_comparison_rejects_empty_rates():
    with pytest.raises(ValueError):
        compare_sample_size_scenarios(0.10, [])
