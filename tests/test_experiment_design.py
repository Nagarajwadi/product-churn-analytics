import pytest

from src.experiment_design import calculate_sample_size


def test_sample_size_matches_expected_example():
    result = calculate_sample_size(
        baseline_rate=0.10,
        treatment_rate=0.08,
        alpha=0.05,
        power=0.80,
    )

    assert result == 3213


def test_sample_size_increases_for_smaller_effect():
    larger_effect = calculate_sample_size(0.10, 0.08)
    smaller_effect = calculate_sample_size(0.10, 0.09)

    assert smaller_effect > larger_effect


def test_sample_size_rejects_invalid_baseline():
    with pytest.raises(ValueError):
        calculate_sample_size(0.0, 0.08)


def test_sample_size_rejects_equal_rates():
    with pytest.raises(ValueError):
        calculate_sample_size(0.10, 0.10)
