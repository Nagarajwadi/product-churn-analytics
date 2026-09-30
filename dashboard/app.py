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

feature_importance = pd.DataFrame({
    "feature": [
        "subscription_count",
        "days_since_last_product_view",
        "days_since_last_login",
        "days_since_last_activity",
        "purchase_rate",
        "purchase_count",
        "total_events",
        "sessions_per_active_day",
        "cart_rate",
        "total_sessions"
    ],
    "importance": [
        0.388617,
        0.114941,
        0.112845,
        0.081866,
        0.077922,
        0.045985,
        0.031120,
        0.018541,
        0.017503,
        0.017250
    ]
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=True
)

st.bar_chart(
    feature_importance.set_index("feature")
)