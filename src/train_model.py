import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


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
    "login_count",
    "product_view_count",
    "search_count",
    "add_to_cart_count",
    "purchase_count",
    "subscription_count",
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


# --------------------------------------------------
# Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# Evaluation
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