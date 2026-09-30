import sqlite3
import pandas as pd


DATABASE_FILE = "data/product_analytics.db"
SQL_FILE = "sql/product_metrics.sql"
OUTPUT_FILE = "data/processed/ml_dataset.csv"


# --------------------------------------------------
# Connect to database
# --------------------------------------------------

connection = sqlite3.connect(DATABASE_FILE)


# --------------------------------------------------
# Read SQL query
# --------------------------------------------------

with open(SQL_FILE, "r") as file:
    query = file.read()


# --------------------------------------------------
# Create ML dataset
# --------------------------------------------------

print("Creating ML dataset...")

df = pd.read_sql_query(
    query,
    connection
)


# --------------------------------------------------
# Save dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("===================================")
print("ML DATASET CREATED")
print("===================================")

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print()
print("Columns:")
print(list(df.columns))

print()
print("Churn distribution:")
print(df["churn"].value_counts())

print()
print(f"Saved to: {OUTPUT_FILE}")


connection.close()