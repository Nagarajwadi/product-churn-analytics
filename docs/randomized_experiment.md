# Randomized A/B Experiment Simulation

## Objective
Simulate a randomized experiment among subscribers and evaluate a synthetic churn outcome.

## Method
- Eligible users: subscribers only.
- Assignment: approximately 50% control and 50% treatment.
- Random seed: 42 for reproducibility.
- Baseline balance: standardized mean differences (SMDs).
- Outcome analysis: risk difference, 95% confidence interval, and two-sided two-proportion z-test.

## Results
The simulation assigned 1,062 subscribers, with 531 in each group.
The largest absolute baseline SMD was 0.067, below the commonly used 0.10 heuristic.

In the recorded run:
- Control: 42/531 (7.91%) simulated churn.
- Treatment: 48/531 (9.04%) simulated churn.
- Risk difference (treatment minus control): +1.13 percentage points.
- 95% confidence interval: -2.22 to +4.48 percentage points.
- Two-sided p-value: 0.5086.

The observed difference was not statistically significant at the 0.05 level.

## Limitations
- Outcomes are simulated, not observed product behavior.
- The simulation imposes different churn probabilities by design; observed results can differ by chance.
- These results do not establish real-world product impact.
- The confidence interval uses a normal approximation.
