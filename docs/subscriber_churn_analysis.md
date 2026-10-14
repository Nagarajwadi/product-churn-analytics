# Subscriber-Only Churn Analysis

## Business question
Among users who subscribed by April 30, 2026, which observed behaviors are associated with cancellation during May–June 2026?

## Dataset summary
- Total users: 3,307
- Eligible subscribers: 1,062
- Churn events: 90
- Subscriber churn rate: 8.47%

## Model evaluation
- Model: standardized logistic regression with class weighting
- Train/test split: 75%/25%, stratified, random_state=42
- ROC-AUC: 0.716
- Average precision: 0.141
- Test churn prevalence: 8.65%
- Churn recall: 87.0% (20 of 23 test churners detected)
- Churn precision: 17.4%

The model detected 20 churners but also flagged 95 non-churners at a threshold of 0.50. The threshold should be evaluated against the business cost of unnecessary retention outreach.

## Important limitations
- Only 90 churn events exist in the full subscriber population.
- Results come from one train/test split and may vary with another sample.
- Class weighting can affect probability calibration.
- Correlated activity and recency features can produce unstable or counterintuitive coefficients.
- These data are synthetic; associations are not causal evidence or validated real-world predictions.

## How to run
```bash
python src/subscriber_churn_analysis.py
python -m pytest tests api/tests -q
```

## Conclusion
This is an exploratory subscriber-retention model with moderate ranking performance in one held-out split. It demonstrates population definition, feature selection, and imbalanced classification, but is not a validated production model.
