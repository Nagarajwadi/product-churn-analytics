import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_FILE = "data/processed/ml_dataset.csv"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading ML dataset...")

df = pd.read_csv(DATA_FILE)

print(f"Loaded {len(df):,} users.")


# --------------------------------------------------
# Define features and target
# --------------------------------------------------

features = [
    "total_events",
    "total_sessions",
    "active_days",
    "login_count",
    "product_view_count",
    "search_count",
    "add_to_cart_count",
    "purchase_count",
    "subscription_count",
    "events_per_active_day",
    "sessions_per_active_day",
    "search_rate",
    "cart_rate",
    "purchase_rate",
    "days_since_last_activity",
    "days_since_last_login",
    "days_since_last_product_view",
]

X = df[features]
y = df["churn"]


# --------------------------------------------------
# Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("Training users:", len(X_train))
print("Testing users:", len(X_test))


# --------------------------------------------------
# Train model
# --------------------------------------------------

print()
print("Training logistic regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)

# Probability of churn
y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# Classification evaluation
# --------------------------------------------------

print()
print("===================================")
print("MODEL EVALUATION")
print("===================================")

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# Probability-based metrics
# --------------------------------------------------

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)

print()
print("===================================")
print("PROBABILITY METRICS")
print("===================================")

print(
    f"ROC-AUC: {roc_auc:.3f}"
)

print(
    f"PR-AUC:  {pr_auc:.3f}"
)


# --------------------------------------------------
# Feature coefficients
# --------------------------------------------------

coefficients = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_[0]
})

coefficients["absolute_coefficient"] = (
    coefficients["coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "absolute_coefficient",
    ascending=False
)

print()
print("===================================")
print("FEATURE COEFFICIENTS")
print("===================================")

print(
    coefficients[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

# ===================================
# RANDOM FOREST EVALUATION
# ===================================

print("\n===================================")
print("RANDOM FOREST EVALUATION")
print("===================================")

rf_predictions = rf_model.predict(X_test)
rf_probability = rf_model.predict_proba(X_test)[:, 1]

print("\nClassification report:")
print(classification_report(y_test, rf_predictions))

print("Confusion matrix:")
print(confusion_matrix(y_test, rf_predictions))

rf_roc_auc = roc_auc_score(y_test, rf_probability)
rf_pr_auc = average_precision_score(y_test, rf_probability)

print("\n===================================")
print("RANDOM FOREST PROBABILITY METRICS")
print("===================================")

print(f"ROC-AUC: {rf_roc_auc:.3f}")
print(f"PR-AUC:  {rf_pr_auc:.3f}")


# ===================================
# THRESHOLD ANALYSIS
# ===================================

print("\n===================================")
print("RANDOM FOREST THRESHOLD ANALYSIS")
print("===================================")

thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

for threshold in thresholds:

    threshold_predictions = (
        rf_probability >= threshold
    ).astype(int)

    threshold_report = classification_report(
        y_test,
        threshold_predictions,
        output_dict=True,
        zero_division=0
    )

    precision = threshold_report["1"]["precision"]
    recall = threshold_report["1"]["recall"]

    true_positives = (
        ((threshold_predictions == 1) & (y_test == 1))
        .sum()
    )

    false_positives = (
        ((threshold_predictions == 1) & (y_test == 0))
        .sum()
    )

    users_targeted = (
        threshold_predictions == 1
    ).sum()

    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Users targeted: {users_targeted}"
    )

    print(
        f"True churners identified: {true_positives}"
    )

    print(
        f"False positives: {false_positives}"
    )

    print(
        f"Precision: {precision:.3f}"
    )

    print(
        f"Recall: {recall:.3f}"
    )

# ===================================
# RANDOM FOREST FEATURE IMPORTANCE
# ===================================

print("\n===================================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("===================================")

feature_importance = pd.DataFrame({
    "feature": features,
    "importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance.to_string(index=False))



# ===================================
# SAVE THRESHOLD RESULTS
# ===================================

threshold_results = []

for threshold in thresholds:

    threshold_predictions = (
        rf_probability >= threshold
    ).astype(int)

    threshold_report = classification_report(
        y_test,
        threshold_predictions,
        output_dict=True,
        zero_division=0
    )

    true_positives = (
        ((threshold_predictions == 1) & (y_test == 1))
        .sum()
    )

    false_positives = (
        ((threshold_predictions == 1) & (y_test == 0))
        .sum()
    )

    users_targeted = (
        threshold_predictions == 1
    ).sum()

    threshold_results.append({
        "threshold": threshold,
        "users_targeted": users_targeted,
        "true_churners_identified": true_positives,
        "false_positives": false_positives,
        "precision": threshold_report["1"]["precision"],
        "recall": threshold_report["1"]["recall"]
    })

threshold_results_df = pd.DataFrame(threshold_results)

threshold_results_df.to_csv(
    "data/processed/model_results.csv",
    index=False
)

print("\nModel results saved to:")
print("data/processed/model_results.csv")

# ===================================
# SAVE FEATURE IMPORTANCE
# ===================================

feature_importance.to_csv(
    "data/processed/feature_importance.csv",
    index=False
)

print("\nFeature importance saved to:")
print("data/processed/feature_importance.csv")