import streamlit as st
import pandas as pd


# ===================================
# PAGE CONFIGURATION
# ===================================

st.set_page_config(
    page_title="Product Churn Analytics",
    page_icon="📊",
    layout="wide"
)


# ===================================
# LOAD DATA
# ===================================

DATA_PATH = "data/processed/ml_dataset.csv"

df = pd.read_csv(DATA_PATH)


# ===================================
# DASHBOARD HEADER
# ===================================

st.title("📊 Product User Churn Analytics")

st.markdown(
    """
    **Subscription Productivity App**

    Analyze user engagement, churn behavior, and machine-learning
    churn predictions.
    """
)


# ===================================
# KEY METRICS
# ===================================

total_users = len(df)
churned_users = int(df["churn"].sum())
non_churned_users = total_users - churned_users
churn_rate = churned_users / total_users


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Users",
        f"{total_users:,}"
    )

with col2:
    st.metric(
        "Churned Users",
        f"{churned_users:,}"
    )

with col3:
    st.metric(
        "Active Users",
        f"{non_churned_users:,}"
    )

with col4:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2%}"
    )


# ===================================
# DATA PREVIEW
# ===================================

st.subheader("Dataset Overview")

st.write(
    f"The dashboard contains **{total_users:,} users** "
    f"and **{len(df.columns)} analytical features**."
)

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ===================================
# CHURN DISTRIBUTION
# ===================================

st.subheader("Churn Distribution")

churn_distribution = (
    df["churn"]
    .value_counts()
    .rename(index={
        0: "Non-Churned",
        1: "Churned"
    })
)

st.bar_chart(churn_distribution)


# ===================================
# ENGAGEMENT BY CHURN STATUS
# ===================================

st.subheader("Average Engagement by Churn Status")

engagement_summary = (
    df.groupby("churn")[
        [
            "total_events",
            "total_sessions",
            "active_days",
            "days_since_last_activity"
        ]
    ]
    .mean()
    .rename(index={
        0: "Non-Churned",
        1: "Churned"
    })
)

st.dataframe(
    engagement_summary.round(2),
    use_container_width=True
)


# ===================================
# ML CHURN RISK
# ===================================

st.subheader("🤖 Machine Learning Churn Risk")

MODEL_RESULTS_PATH = "data/processed/model_results.csv"

model_results = pd.read_csv(MODEL_RESULTS_PATH)

st.markdown(
    """
    The Random Forest model ranks users by estimated churn risk.
    Adjust the threshold to explore the trade-off between identifying
    potential churners and targeting additional users.
    """
)

threshold = st.slider(
    "Prediction Threshold",
    min_value=0.20,
    max_value=0.60,
    value=0.50,
    step=0.10
)

selected_result = model_results[
    model_results["threshold"].round(2) == round(threshold, 2)
]

if not selected_result.empty:

    result = selected_result.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Users Targeted",
            f"{int(result['users_targeted']):,}"
        )

    with col2:
        st.metric(
            "Churners Identified",
            f"{int(result['true_churners_identified']):,}"
        )

    with col3:
        st.metric(
            "False Positives",
            f"{int(result['false_positives']):,}"
        )

    with col4:
        st.metric(
            "Recall",
            f"{result['recall']:.1%}"
        )

    st.write(
        f"**Precision:** {result['precision']:.1%}"
    )

else:

    st.warning(
        "No threshold result available for the selected threshold."
    )


# ===================================
# MODEL PERFORMANCE
# ===================================

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Random Forest ROC-AUC",
        "0.946"
    )

with col2:
    st.metric(
        "Random Forest PR-AUC",
        "0.383"
    )


# ===================================
# FEATURE IMPORTANCE
# ===================================

st.subheader("🔍 Churn Prediction Feature Importance")

FEATURE_IMPORTANCE_PATH = (
    "data/processed/feature_importance.csv"
)

feature_importance = pd.read_csv(
    FEATURE_IMPORTANCE_PATH
)

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=True
)

st.bar_chart(
    feature_importance.set_index("feature")
)

st.caption(
    "Feature importance shows how much each feature contributed "
    "to the Random Forest's predictions. It does not indicate "
    "whether a feature increases or decreases churn."
)


# ===================================
# LOAD USER CHURN PREDICTIONS
# ===================================

PREDICTIONS_PATH = (
    "data/processed/user_churn_predictions.csv"
)

user_predictions = pd.read_csv(
    PREDICTIONS_PATH
)


# ===================================
# USER RISK SUMMARY
# ===================================

st.subheader("📌 Churn Risk Summary")

high_risk_users = (
    user_predictions["risk_segment"] == "High"
).sum()

medium_risk_users = (
    user_predictions["risk_segment"] == "Medium"
).sum()

low_risk_users = (
    user_predictions["risk_segment"] == "Low"
).sum()

high_probability_users = (
    user_predictions["churn_probability"] >= 0.50
).sum()

risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)

with risk_col1:
    st.metric(
        "🔴 High Risk",
        high_risk_users
    )

with risk_col2:
    st.metric(
        "🟡 Medium Risk",
        medium_risk_users
    )

with risk_col3:
    st.metric(
        "🟢 Low Risk",
        low_risk_users
    )

with risk_col4:
    st.metric(
        "🎯 Probability ≥ 50%",
        high_probability_users
    )


# ===================================
# RISK SEGMENT DISTRIBUTION
# ===================================

st.subheader("📊 Risk Segment Distribution")

risk_distribution = (
    user_predictions["risk_segment"]
    .value_counts()
    .reindex(["High", "Medium", "Low"])
    .fillna(0)
)

st.bar_chart(risk_distribution)

st.caption(
    "Distribution of predicted churn risk among the model's "
    "held-out test users."
)

# ===================================
# USER CHURN RISK TABLE
# ===================================


risk_filter = st.selectbox(
    "Risk Segment",
    ["All", "High", "Medium", "Low"]
)

if risk_filter != "All":
    filtered_predictions = user_predictions[
        user_predictions["risk_segment"] == risk_filter
    ]
else:
    filtered_predictions = user_predictions

display_predictions = filtered_predictions.copy()

display_predictions["churn_probability"] = (
    display_predictions["churn_probability"] * 100
).round(1)

display_predictions = display_predictions.rename(
    columns={
        "user_id": "User ID",
        "actual_churn": "Actual Churn",
        "churn_probability": "Churn Probability (%)",
        "risk_segment": "Risk Segment"
    }
)

st.dataframe(
    display_predictions[
        [
            "User ID",
            "Churn Probability (%)",
            "Risk Segment",
            "Actual Churn"
        ]
    ],
    use_container_width=True
)