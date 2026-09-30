import sqlite3
import pandas as pd


DATABASE_FILE = "data/product_analytics.db"
SQL_FILE = "sql/product_metrics.sql"


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
# Execute query
# --------------------------------------------------

print("Running SQL query...")

result = pd.read_sql_query(
    query,
    connection
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print()
print("===================================")
print("PRODUCT METRICS")
print("===================================")

print(result.head(20).to_string(index=False))

print()
print(f"Users returned: {len(result):,}")


# --------------------------------------------------
# Close connection
# --------------------------------------------------

connection.close()
