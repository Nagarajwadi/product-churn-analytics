import sqlite3
import sys
from pathlib import Path

import pandas as pd

DATABASE_FILE = Path("data/product_analytics.db")
DEFAULT_SQL_FILE = Path("sql/product_metrics.sql")


def main():
    """Execute a SQL file against the product analytics database."""
    sql_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SQL_FILE

    if not sql_file.is_file():
        raise FileNotFoundError(f"SQL file not found: {sql_file}")

    if not DATABASE_FILE.is_file():
        raise FileNotFoundError(f"Database not found: {DATABASE_FILE}")

    query = sql_file.read_text(encoding="utf-8")

    print(f"Running SQL query: {sql_file}")

    with sqlite3.connect(DATABASE_FILE) as connection:
        result = pd.read_sql_query(query, connection)

    print()
    print("=" * 50)
    print(f"QUERY RESULTS: {sql_file.name}")
    print("=" * 50)
    print(result.head(20).to_string(index=False))
    print()
    print(f"Rows returned: {len(result):,}")


if __name__ == "__main__":
    main()
