import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


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
# EXECUTIVE SUMMARY
# ===================================

st.markdown("---")
st.header("📋 Executive Summary")

st.markdown(
    """
    This project analyzes user behavior and predicts churn risk to help
    product teams prioritize customer-retention efforts.

    **Key findings**

    - **Churn prediction:** The Random Forest model achieved a ROC-AUC
      of 0.953 and average precision of 0.402 on the held-out test set.
    - **Risk identification:** At the 0.50 classification threshold,
      the model identified 16 of 18 actual churners, with 97 false
      positives among 113 flagged users.
    - **Campaign planning:** The ROI calculator estimates possible
      campaign economics using adjustable cost, retention-uplift, and
      contribution assumptions.
    - **Experimentation:** The A/B test section demonstrates statistical
      testing using simulated groups, not a real retention experiment.

    **Recommended business priorities**

    1. Review high-risk users and prioritize appropriate, low-cost
       retention outreach.
    2. Evaluate outreach cost and potential contribution before
       committing to expensive incentives.
    3. Run a properly randomized retention experiment and measure
       actual retention, conversion, and incremental contribution.
    4. Reassess the model as more churn outcomes and campaign results
       become available.

    **Important limitation:** Model performance and ROI scenarios do
    not prove that a retention campaign will prevent churn. Validate
    campaign impact with real experimental outcomes.
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
# INTERACTIVE USER EXPLORER
# ===================================

st.subheader("Explore User Segments")

col1, col2 = st.columns(2)

with col1:
    churn_filter = st.selectbox(
        "Churn Status",
        options=["All Users", "Non-Churned", "Churned"]
    )

with col2:
    max_recency = st.slider(
        "Maximum Days Since Last Activity",
        min_value=0,
        max_value=max(1, int(df["days_since_last_activity"].max())),
        value=int(df["days_since_last_activity"].max())
    )

filtered_df = df.copy()

if churn_filter == "Non-Churned":
    filtered_df = filtered_df[filtered_df["churn"] == 0]
elif churn_filter == "Churned":
    filtered_df = filtered_df[filtered_df["churn"] == 1]

filtered_df = filtered_df[
    filtered_df["days_since_last_activity"] <= max_recency
]

st.write(f"**Users matching filters:** {len(filtered_df):,}")

st.dataframe(
    filtered_df.head(100),
    width="stretch"
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download filtered users as CSV",
    data=csv_data,
    file_name="filtered_churn_users.csv",
    mime="text/csv"
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
    width="stretch"
)


# ===================================
# CHURN BEHAVIOR COMPARISON CHARTS
# ===================================

st.subheader("Churn Behavior Comparison")

chart_data = (
    df.groupby("churn")[
        [
            "active_days",
            "total_sessions",
            "days_since_last_activity",
            "days_since_last_login",
        ]
    ]
    .mean()
    .rename(index={
        0: "Non-Churned",
        1: "Churned",
    })
    .rename(columns={
        "active_days": "Active Days",
        "total_sessions": "Total Sessions",
        "days_since_last_activity": "Days Since Activity",
        "days_since_last_login": "Days Since Login",
    })
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average Engagement**")
    st.bar_chart(
        chart_data[["Active Days", "Total Sessions"]]
    )

with col2:
    st.markdown("**Average Inactivity**")
    st.bar_chart(
        chart_data[["Days Since Activity", "Days Since Login"]]
    )



# ===================================
# CHURN-RISK PRIORITIZATION
# ===================================

st.subheader("🎯 Churn-Risk Prioritization")

PREDICTIONS_PATH = "data/processed/user_churn_predictions.csv"
predictions = pd.read_csv(PREDICTIONS_PATH)

risk_counts = predictions["risk_segment"].value_counts()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("High-Risk Users", int(risk_counts.get("High", 0)))

with col2:
    st.metric("Medium-Risk Users", int(risk_counts.get("Medium", 0)))

with col3:
    st.metric("Low-Risk Users", int(risk_counts.get("Low", 0)))

risk_order = ["High", "Medium", "Low"]

col1, col2 = st.columns(2)

with col1:
    selected_risk = st.multiselect(
        "Risk Segment",
        options=risk_order,
        default=["High", "Medium", "Low"],
    )

with col2:
    min_probability = st.slider(
        "Minimum Predicted Churn Probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f",
    )

filtered_predictions = predictions[
    predictions["risk_segment"].isin(selected_risk)
    & (predictions["churn_probability"] >= min_probability)
].copy()

filtered_predictions = filtered_predictions.sort_values(
    "churn_probability",
    ascending=False,
)

filtered_predictions["churn_probability"] = (
    filtered_predictions["churn_probability"].map(
        lambda probability: f"{probability:.1%}"
    )
)

st.write(f"**Users matching filters:** {len(filtered_predictions):,}")

st.dataframe(
    filtered_predictions[
        ["user_id", "risk_segment", "churn_probability", "actual_churn"]
    ],
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download churn-risk table as CSV",
    data=filtered_predictions.to_csv(index=False).encode("utf-8"),
    file_name="churn_risk_prioritization.csv",
    mime="text/csv",
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
# MODEL PERFORMANCE AND EVALUATION
# ===================================

st.subheader("Model Performance")

EVALUATION_PATH = "data/processed/model_evaluation.csv"
CONFUSION_MATRIX_PATH = "data/processed/confusion_matrix.csv"

evaluation = pd.read_csv(EVALUATION_PATH, index_col="class")
confusion = pd.read_csv(CONFUSION_MATRIX_PATH, index_col="actual")

# Probability-based metrics
col1, col2 = st.columns(2)

with col1:
    st.metric("Random Forest ROC-AUC", "0.953")

with col2:
    st.metric("Random Forest PR-AUC", "0.402")

# Classification metrics for the churn class
st.markdown("### Churn Detection Metrics")

churn_metrics = evaluation.loc["1"]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Precision", f"{churn_metrics['precision']:.1%}")

with col2:
    st.metric("Recall", f"{churn_metrics['recall']:.1%}")

with col3:
    st.metric("F1-score", f"{churn_metrics['f1-score']:.3f}")

st.caption(
    "Metrics are calculated on the held-out test set. "
    "Precision measures how many flagged users actually churned; "
    "recall measures how many actual churners were identified."
)

# Confusion matrix
st.markdown("### Confusion Matrix")

st.dataframe(confusion, width="stretch")

st.caption(
    "Rows represent actual outcomes; columns represent predicted outcomes."
)



# ===================================
# PRECISION-RECALL CURVE
# ===================================

st.markdown("### Precision–Recall Curve")

PREDICTIONS_PATH = "data/processed/user_churn_predictions.csv"
pr_data = pd.read_csv(PREDICTIONS_PATH)

precision_values, recall_values, _ = precision_recall_curve(
    pr_data["actual_churn"],
    pr_data["churn_probability"]
)

pr_auc_value = average_precision_score(
    pr_data["actual_churn"],
    pr_data["churn_probability"]
)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    recall_values,
    precision_values,
    label=f"Average precision = {pr_auc_value:.3f}"
)

baseline = pr_data["actual_churn"].mean()
ax.axhline(
    y=baseline,
    linestyle="--",
    label=f"Churn prevalence = {baseline:.1%}"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Each point represents a probability threshold. "
    "Higher recall catches more churners, while higher precision "
    "reduces unnecessary retention outreach."
)


# ===================================
# BUSINESS INTERPRETATION
# ===================================

st.markdown("### Business Interpretation & Recommended Actions")

st.markdown(
    """
    **What the results mean**

    - **High recall (88.9%):** The model identifies 16 of the 18 actual
      churners in this test set.
    - **Low precision (14.2%):** Of the 113 users flagged at the 0.50
      threshold, 16 actually churned and 97 did not.
    - **Imbalanced data:** Only 18 of the 662 test users churned, so
      accuracy alone would not tell the full story.

    **Recommended product actions**

    1. Test low-cost, targeted retention messages for high-risk users.
    2. Reserve expensive incentives for users who meet additional
       business criteria.
    3. Compare retention and conversion outcomes against a control
       group before claiming the campaign reduced churn.
    4. Monitor precision, recall, and campaign cost as more outcomes
       become available.

    **Important limitation:** These are retrospective test-set results.
    They do not prove that contacting a flagged user will prevent churn.
    """
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

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(
    feature_importance["feature"],
    feature_importance["importance"]
)
ax.set_xlabel("Importance")
ax.set_ylabel("Feature")
plt.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Feature importance shows how much each feature contributed "
    "to the Random Forest's predictions. It does not indicate "
    "whether a feature increases or decreases churn."
)



# ===================================
# SIMULATED A/B TEST ANALYSIS
# ===================================

st.markdown("---")
st.subheader("🧪 A/B Test Analysis")

st.warning(
    "**Simulated demonstration only:** The dataset does not contain "
    "a real experiment assignment or intervention. These groups were "
    "randomly created from historical churn labels. The results do not "
    "measure the causal effect of a retention campaign."
)

AB_TEST_SUMMARY_PATH = "data/processed/ab_test_group_summary.csv"
AB_TEST_RESULTS_PATH = "data/processed/ab_test_results.csv"

if (
    os.path.exists(AB_TEST_SUMMARY_PATH)
    and os.path.exists(AB_TEST_RESULTS_PATH)
):
    ab_summary = pd.read_csv(AB_TEST_SUMMARY_PATH)
    ab_results = pd.read_csv(AB_TEST_RESULTS_PATH)

    if (
        not ab_summary.empty
        and not ab_results.empty
        and {"experiment_group", "users", "churners", "churn_rate_pct"}
        .issubset(ab_summary.columns)
    ):
        ab_result = ab_results.iloc[0]

        control_row = ab_summary[
            ab_summary["experiment_group"] == "control"
        ]
        treatment_row = ab_summary[
            ab_summary["experiment_group"] == "treatment"
        ]

        if not control_row.empty and not treatment_row.empty:
            control_row = control_row.iloc[0]
            treatment_row = treatment_row.iloc[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Control Churn Rate",
                    f"{control_row['churn_rate_pct']:.2f}%",
                    help=(
                        f"{int(control_row['churners'])} churners out of "
                        f"{int(control_row['users'])} users"
                    ),
                )

            with col2:
                st.metric(
                    "Treatment Churn Rate",
                    f"{treatment_row['churn_rate_pct']:.2f}%",
                    delta=(
                        f"{ab_result['absolute_difference_percentage_points']:+.3f} "
                        "percentage points"
                    ),
                    delta_color="inverse",
                    help=(
                        f"{int(treatment_row['churners'])} churners out of "
                        f"{int(treatment_row['users'])} users. "
                        "Negative difference means lower observed churn."
                    ),
                )

            st.markdown("#### Group Comparison")

            chart_data = ab_summary.set_index(
                "experiment_group"
            )[["churn_rate_pct"]]

            st.bar_chart(chart_data)

            st.markdown("#### Statistical Test")

            metric1, metric2, metric3 = st.columns(3)

            with metric1:
                st.metric(
                    "P-value",
                    f"{ab_result['p_value']:.4f}",
                )

            with metric2:
                st.metric(
                    "Relative Difference",
                    f"{ab_result['relative_difference_pct']:+.2f}%",
                )

            with metric3:
                st.metric(
                    "Confidence Level",
                    f"{ab_result['confidence_level']:.0%}",
                )

            st.write(
                "95% confidence interval for the absolute churn-rate "
                "difference (treatment minus control): "
                f"**{ab_result['difference_ci_lower_percentage_points']:+.3f} "
                "to "
                f"{ab_result['difference_ci_upper_percentage_points']:+.3f} "
                "percentage points**."
            )

            if ab_result["p_value"] < 0.05:
                st.info(
                    "The simulated groups differ at the 5% significance "
                    "level. This is not evidence of a treatment effect."
                )
            else:
                st.info(
                    "The difference between the simulated groups is not "
                    "statistically significant at the 5% level."
                )

            st.caption(
                "To evaluate a real intervention, assign eligible users "
                "randomly before the intervention, record their actual "
                "group assignments, and measure outcomes over a defined "
                "follow-up period."
            )
        else:
            st.error(
                "The A/B test summary must contain both control and "
                "treatment groups. Run src/ab_test_analysis.py again."
            )
    else:
        st.error(
            "The A/B test output files are empty or have an unexpected "
            "schema. Run src/ab_test_analysis.py again."
        )
else:
    st.info(
        "A/B test results are not available. Run "
        "`python src/ab_test_analysis.py` to generate them."
    )



# ===================================
# RETENTION CAMPAIGN ROI ANALYSIS
# ===================================

st.markdown("---")
st.subheader("💰 Retention Campaign ROI Analysis")

st.warning(
    "Planning estimates only. Campaign response, retention uplift, "
    "and contribution margin are assumptions—not measured outcomes. "
    "Predicted churn risk does not prove that a user will respond."
)

ROI_PREDICTIONS_PATH = "data/processed/user_churn_predictions.csv"

if os.path.exists(ROI_PREDICTIONS_PATH):
    roi_predictions = pd.read_csv(ROI_PREDICTIONS_PATH)

    required_roi_columns = {"user_id", "churn_probability", "risk_segment"}

    if required_roi_columns.issubset(roi_predictions.columns):
        roi_predictions = roi_predictions.drop_duplicates(
            subset=["user_id"]
        )

        risk_options = ["High", "Medium", "Low", "All users"]

        selected_risk = st.selectbox(
            "Target audience",
            risk_options,
            index=0,
            key="roi_target_audience",
        )

        if selected_risk == "All users":
            eligible_users = roi_predictions.copy()
        else:
            eligible_users = roi_predictions[
                roi_predictions["risk_segment"] == selected_risk
            ].copy()

        st.caption(
            f"Eligible audience in the current prediction file: "
            f"{len(eligible_users):,} users."
        )

        input_col1, input_col2, input_col3 = st.columns(3)

        with input_col1:
            campaign_cost_per_user = st.number_input(
                "Campaign cost per contacted user (₹)",
                min_value=0.0,
                value=20.0,
                step=5.0,
                key="roi_cost_per_user",
            )

        with input_col2:
            assumed_retention_uplift = st.number_input(
                "Assumed incremental retention (%)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=1.0,
                key="roi_retention_uplift",
                help=(
                    "Hypothetical percentage-point increase in retention "
                    "among contacted users, not a measured campaign result."
                ),
            )

        with input_col3:
            contribution_per_retained_user = st.number_input(
                "Contribution per incremental retained user (₹)",
                min_value=0.0,
                value=500.0,
                step=50.0,
                key="roi_contribution",
                help=(
                    "Estimated contribution after variable costs, not "
                    "gross revenue."
                ),
            )

        audience_size = len(eligible_users)
        campaign_cost = audience_size * campaign_cost_per_user

        # This is an assumption-based scenario, not a churn-model
        # prediction or an experimentally measured treatment effect.
        expected_incremental_retained = (
            audience_size * assumed_retention_uplift / 100
        )

        estimated_incremental_contribution = (
            expected_incremental_retained
            * contribution_per_retained_user
        )

        net_benefit = estimated_incremental_contribution - campaign_cost

        roi_pct = (
            net_benefit / campaign_cost * 100
            if campaign_cost > 0
            else None
        )

        break_even_uplift_pct = (
            campaign_cost
            / (audience_size * contribution_per_retained_user)
            * 100
            if audience_size > 0 and contribution_per_retained_user > 0
            else None
        )

        st.markdown("#### Scenario results")

        result_col1, result_col2, result_col3, result_col4 = st.columns(4)

        result_col1.metric(
            "Estimated campaign cost",
            f"₹{campaign_cost:,.2f}",
        )

        result_col2.metric(
            "Incremental retained users",
            f"{expected_incremental_retained:,.1f}",
        )

        result_col3.metric(
            "Estimated net benefit",
            f"₹{net_benefit:,.2f}",
        )

        result_col4.metric(
            "Estimated ROI",
            f"{roi_pct:,.1f}%" if roi_pct is not None else "N/A",
        )

        if break_even_uplift_pct is not None:
            st.write(
                "Break-even incremental retention uplift: "
                f"**{break_even_uplift_pct:.2f}%** under these assumptions."
            )
        else:
            st.info(
                "Break-even uplift cannot be calculated when the audience "
                "or contribution per retained user is zero."
            )

        roi_chart = pd.DataFrame(
            {
                "Amount (₹)": [
                    campaign_cost,
                    estimated_incremental_contribution,
                    net_benefit,
                ]
            },
            index=[
                "Campaign cost",
                "Estimated contribution",
                "Estimated net benefit",
            ],
        )

        st.markdown("#### Estimated financial impact")
        st.bar_chart(roi_chart)

        st.caption(
            "Scenario formula: net benefit = estimated incremental "
            "contribution − campaign cost. ROI = net benefit ÷ campaign "
            "cost × 100. Validate these assumptions with a properly "
            "randomized experiment before treating them as business results."
        )
    else:
        st.error(
            "Prediction data is missing required columns for ROI analysis."
        )
else:
    st.info(
        "Prediction data is unavailable. Run the model training script "
        "to generate data/processed/user_churn_predictions.csv."
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
        "🎯 Risk Score ≥ 50%",
        high_probability_users
    )


# ===================================
# RISK SEGMENT DISTRIBUTION
# ===================================

st.subheader("📊 Risk Segment Distribution")

risk_distribution = (
    user_predictions["risk_segment"]
    .value_counts()
    .reindex(["Low", "Medium", "High"])
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    risk_distribution.index,
    risk_distribution.values
)

ax.set_ylabel("Users")
ax.set_xlabel("Risk Segment")
ax.set_title("Predicted Churn Risk Distribution")

for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{int(height):,}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Distribution of predicted churn risk among the model's "
    "held-out test users."
)

# ===================================
# PRODUCT INSIGHTS
# ===================================

st.subheader("💡 Product Insights")

high_risk_pct = (
    high_risk_users / len(user_predictions)
)

churned_engagement = engagement_summary.loc[
    "Churned"
]

non_churned_engagement = engagement_summary.loc[
    "Non-Churned"
]

st.markdown(
    f"""
    **Risk concentration:** {high_risk_users:,} of
    {len(user_predictions):,} evaluated users
    ({high_risk_pct:.1%}) are classified as high risk.

    **Observed engagement:** Churned users averaged
    **{churned_engagement["active_days"]:.1f} active days**
    compared with **{non_churned_engagement["active_days"]:.1f}**
    for non-churned users.

    **Recent activity:** Churned users averaged
    **{churned_engagement["days_since_last_activity"]:.1f} days**
    since their last activity, compared with
    **{non_churned_engagement["days_since_last_activity"]:.1f} days**
    for non-churned users.

    **Model signal:** The Random Forest identifies
    user subscription history and activity recency among
    the features contributing most to its predictions.
    """
)

# ===================================
# USER BEHAVIORAL CONTEXT
# ===================================

st.subheader("🔎 User Behavioral Context")

selected_user = st.selectbox(
    "Select a User ID",
    user_predictions["user_id"].tolist()
)

selected_user_data = df[
    df["user_id"] == selected_user
]

selected_prediction = user_predictions[
    user_predictions["user_id"] == selected_user
]

if not selected_prediction.empty:

    prediction_row = selected_prediction.iloc[0]

    st.markdown("### 🎯 User Risk Profile")

    profile_col1, profile_col2 = st.columns(2)

    with profile_col1:
        st.metric(
            "Churn Risk Score",
            f"{prediction_row['churn_probability']:.1%}"
        )

    with profile_col2:
        st.metric(
            "Risk Segment",
            prediction_row["risk_segment"]
        )
if not selected_user_data.empty:

    user_row = selected_user_data.iloc[0]

    behavior_col1, behavior_col2, behavior_col3, behavior_col4 = st.columns(4)

    with behavior_col1:
        st.metric(
            "Total Events",
            f"{int(user_row['total_events']):,}"
        )

    with behavior_col2:
        st.metric(
            "Active Days",
            f"{int(user_row['active_days']):,}"
        )

    with behavior_col3:
        st.metric(
            "Login Count",
            f"{int(user_row['login_count']):,}"
        )

    with behavior_col4:
        st.metric(
            "Days Since Activity",
            f"{int(user_row['days_since_last_activity']):,}"
        )

    behavior_details = pd.DataFrame({
        "Metric": [
            "Total Sessions",
            "Product Views",
            "Searches",
            "Add to Cart",
            "Purchases",
            "Subscriptions",
            "Days Since Last Login",
            "Days Since Last Product View"
        ],
        "Value": [
            user_row["total_sessions"],
            user_row["product_view_count"],
            user_row["search_count"],
            user_row["add_to_cart_count"],
            user_row["purchase_count"],
            user_row["subscription_count"],
            user_row["days_since_last_login"],
            user_row["days_since_last_product_view"]
        ]
    })

    st.dataframe(
        behavior_details,
        hide_index=True,
        width="stretch"
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
        "churn_probability": "Churn Risk Score (%)",
        "risk_segment": "Risk Segment"
    }
)

st.dataframe(
    display_predictions[
        [
            "User ID",
            "Churn Risk Score (%)",
            "Risk Segment",
        ]
    ],
    width="stretch"
)