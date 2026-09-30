import sqlite3
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CSV_FILE = "data/processed/clean_events.csv"
DATABASE_FILE = "data/product_analytics.db"


# --------------------------------------------------
# Load CSV
# --------------------------------------------------

print("Loading clean event data...")

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df):,} events.")


# --------------------------------------------------
# Create SQLite database
# --------------------------------------------------

print()
print("Creating SQLite database...")

connection = sqlite3.connect(DATABASE_FILE)


# --------------------------------------------------
# Write data to database
# --------------------------------------------------

df.to_sql(
    "clean_events",
    connection,
    if_exists="replace",
    index=False
)


# --------------------------------------------------
# Verify
# --------------------------------------------------

cursor = connection.cursor()

cursor.execute(
    "SELECT COUNT(*) FROM clean_events"
)

row_count = cursor.fetchone()[0]

print(
    f"Database contains {row_count:,} events."
)


# --------------------------------------------------
# Close connection
# --------------------------------------------------

connection.close()

print()
print("Database setup complete.")
print(f"Saved to: {DATABASE_FILE}")
