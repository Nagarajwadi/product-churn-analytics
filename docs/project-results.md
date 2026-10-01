# Product User Churn Prediction & Analytics

## Project Overview

This project demonstrates an end-to-end product analytics and machine learning workflow for identifying users at risk of churn in a fictional subscription-based productivity application.

The project combines **Python, SQL, ETL, exploratory data analysis, machine learning, product analytics, and Streamlit dashboarding** to transform raw product events into actionable churn-risk insights.

### Business Problem

Subscription products need to identify users who may be at risk of cancelling so that product and customer-success teams can prioritize potential retention interventions.

The objective of this project is to:

- Transform raw product event data into an analysis-ready dataset
- Define measurable product engagement and behavioral features
- Identify patterns associated with observed churn
- Build models that estimate user-level churn risk
- Evaluate the trade-offs between precision and recall
- Translate model outputs into a product-facing dashboard
- Demonstrate how analytical results could support retention prioritization

The dataset is synthetic and is intended to demonstrate the analytical workflow rather than represent real customer behavior.


## Data & ETL Pipeline

The project uses synthetic product event data representing user activity in a subscription-based productivity application.

### Dataset

The event data contains:

- **5,000 users**
- Product events such as signups, logins, product views, searches, cart actions, purchases, subscriptions, and cancellations
- User attributes including country, device, and subscription plan
- Event timestamps covering January-June 2026

The ETL pipeline performs the following steps:

1. **Extract** — reads the raw product event CSV.
2. **Transform** — parses timestamps, validates event types, removes duplicates, sorts events chronologically, and checks required fields.
3. **Load** — writes the cleaned events to `data/processed/clean_events.csv` and loads them into SQLite.

The final cleaned dataset contains **229,762 validated events** with no duplicate rows or missing required values.

## Churn Definition & Feature Engineering

To avoid using future information when creating model features, the analysis separates the data into two time periods:

- **Feature window:** January-April 2026
- **Outcome window:** May-June 2026

A user is labelled as churned when a `cancel_subscription` event occurs during the outcome window.

The feature dataset contains **3,307 eligible users** and **17 predictive features**.

### Feature categories

**Engagement**

- Total events
- Total sessions
- Active days
- Login count
- Product views
- Searches
- Add-to-cart events
- Purchases
- Subscription count

**Recency**

- Days since last activity
- Days since last login
- Days since last product view

**Behavioral rates**

- Events per active day
- Sessions per active day
- Search rate
- Cart rate
- Purchase rate

These features provide both absolute activity measures and normalized behavioral signals for modeling.


## Exploratory Data Analysis

The exploratory analysis examined user engagement, activity recency, behavioral rates, and relationships with observed churn.

### Key EDA findings

- The dataset contains **3,307 eligible users**, of whom **90 churned**, resulting in a **2.72% churn rate**.
- Churn is highly imbalanced, making accuracy alone unsuitable as the primary evaluation metric.
- Churned users show higher average raw activity counts across several engagement measures in this synthetic dataset.
- Normalized behavioral rates are much closer between churned and non-churned users than raw activity volumes.
- Churned users have lower average values for days since last activity, login, and product view in this synthetic dataset.
- Subscription history and purchase activity show relatively strong pairwise associations with the churn label.
- Correlation analysis identifies linear associations but does not establish causality.

The detailed exploratory analysis is available in:

`notebooks/01_product_churn_eda.ipynb`

## Machine Learning & Model Evaluation

Two classification approaches were evaluated:

- **Logistic Regression**
- **Random Forest**

The dataset was split into an **80% training set and 20% test set** using stratified sampling.

Because only **2.72%** of users churned, the analysis emphasizes **ROC-AUC, PR-AUC, precision, and recall** rather than accuracy alone.

### Model Performance

| Model | ROC-AUC | PR-AUC | Churn Precision | Churn Recall | Churn F1 |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.931 | 0.168 | 13.1% | 100.0% | 0.232 |
| Random Forest | 0.953 | 0.402 | 14.2% | 88.9% | 0.244 |

On this held-out synthetic test set, the Random Forest produced higher ROC-AUC and PR-AUC, while Logistic Regression achieved higher recall at the default 0.50 classification threshold.

The results demonstrate why multiple evaluation metrics are useful for an imbalanced churn problem.


## Churn Risk Threshold Analysis

The Random Forest produces a probability of churn for each evaluated user. Rather than treating 0.50 as an automatically optimal cutoff, several probability thresholds were examined.

| Threshold | Users Targeted | True Churners Identified | False Positives | Precision | Recall |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 137 | 17 | 120 | 12.4% | 94.4% |
| 0.30 | 129 | 17 | 112 | 13.2% | 94.4% |
| 0.40 | 119 | 16 | 103 | 13.4% | 88.9% |
| 0.50 | 113 | 16 | 97 | 14.2% | 88.9% |
| 0.60 | 99 | 16 | 83 | 16.2% | 88.9% |

The analysis shows that increasing the threshold reduces the number of users targeted and increases precision, while recall remains lower at the higher thresholds.

There is no universally correct threshold. The appropriate operating point would depend on factors such as retention-team capacity and the relative business cost of false positives versus missed churners.

## Model Feature Importance

The Random Forest feature importance analysis identified the following leading predictive signals:

| Feature | Importance |
|---|---:|
| subscription_count | 0.3945 |
| days_since_last_login | 0.1165 |
| days_since_last_product_view | 0.0989 |
| days_since_last_activity | 0.0857 |
| purchase_rate | 0.0814 |
| purchase_count | 0.0442 |

Activity recency and purchase-related features contribute substantially to the model's predictive splits.

`subscription_count` has the highest importance, but this result requires particular caution. The synthetic churn definition requires a subscription before a cancellation event can occur, so this relationship is partly created by the dataset design. It should not be interpreted as evidence that subscription history causes churn.

Feature importance indicates contribution to Random Forest predictions and does not establish the direction or causality of a relationship.

### Robustness Check: Removing Subscription History

Because `subscription_count` is closely related to churn eligibility in the synthetic dataset, a controlled experiment was performed by removing this feature and retraining the Random Forest using the same train/test split and model configuration.

| Random Forest configuration | ROC-AUC | PR-AUC |
|---|---:|---:|
| With `subscription_count` | 0.953 | 0.402 |
| Without `subscription_count` | 0.899 | 0.361 |

The model retained substantial predictive signal after removing `subscription_count`, with ROC-AUC of 0.899 and PR-AUC of 0.361. This suggests that behavioral and recency features also contribute meaningful predictive information.

This robustness check does not remove the dataset-design limitation, but it provides additional context when interpreting the model's performance and feature importance.


## Product Analytics Insights

The project translates the model outputs into product-oriented insights rather than treating the model as an isolated machine-learning exercise.

Key analytical outputs include:

- User-level estimated churn probabilities
- High, medium, and low risk segments
- Threshold-based retention targeting scenarios
- Engagement and activity-recency context for individual users
- Model feature importance for interpreting predictive signals

These outputs could support a workflow in which product or customer-success teams prioritize users for further investigation or potential retention interventions.

The analysis should be treated as a prioritization framework rather than an automated decision system.


## Dashboard

The project includes an interactive **Streamlit dashboard** that connects the analytical dataset with the machine-learning outputs.

The dashboard provides:

- Overall user and churn KPIs
- Churn distribution
- Engagement comparison by churn status
- Model-based churn risk probabilities
- High, medium, and low risk segments
- Threshold-based targeting analysis
- Random Forest feature importance
- Individual user behavioral context
- Individual user churn-risk profiles
- A filterable user churn-risk table

The dashboard is designed to demonstrate how analytical and machine-learning outputs can be translated into a product-facing decision-support interface.

Dashboard preview:

![Product Churn Analytics Dashboard](dashboard-overview.png)

## Technical Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Database & Analytics | SQLite, SQL |
| ETL | Python ETL pipeline |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Development | Jupyter, Git, GitHub Codespaces |
| Data Format | CSV |

## Project Limitations

This project is designed as a portfolio demonstration using synthetic data, so several limitations should be considered:

- The event data is synthetic and may not reflect real customer behavior.
- The churn rate and behavioral patterns are determined by the data-generation process.
- The churn outcome is based on a fixed observation window rather than a production retention definition.
- The dataset contains relatively few churned users, which limits statistical power.
- Model performance was evaluated on a single train/test split rather than through production monitoring or repeated cross-validation.
- Random Forest feature importance shows predictive contribution, not causality.
- The strong importance of `subscription_count` is partly influenced by the synthetic churn-generation logic.
- A production implementation would require real customer data, leakage checks, model monitoring, calibration analysis, and business-defined intervention costs.

These limitations are important when interpreting the model results and demonstrate why predictive analytics should be evaluated in the context of both data quality and business requirements.

## Conclusion

This project demonstrates an end-to-end workflow for turning raw product events into actionable churn-risk analysis.

The workflow covers:

**Raw Events → ETL → SQL Analytics → Feature Engineering → EDA → Machine Learning → Model Evaluation → Risk Segmentation → Product Dashboard**

The project demonstrates practical skills across both **data science** and **product analytics**, including data preparation, SQL analysis, behavioral feature engineering, imbalanced classification, model interpretation, threshold analysis, and communicating analytical results through an interactive dashboard.
