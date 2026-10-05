import json
import sqlite3
from pathlib import Path

import pandas as pd


API_FILE = Path("data/raw/api_users.json")
DATABASE_FILE = "data/product_analytics.db"


def load_api_users():
    """Load API user data into the SQLite database."""
    print("Loading API user data...")

    with API_FILE.open("r", encoding="utf-8") as file:
        users = json.load(file)

    df = pd.DataFrame(users)

    print(f"Loaded {len(df):,} API users.")

    connection = sqlite3.connect(DATABASE_FILE)

    df.to_sql(
        "api_users",
        connection,
        if_exists="replace",
        index=False,
    )

    cursor = connection.cursor()

    cursor.execute("SELECT COUNT(*) FROM api_users")
    row_count = cursor.fetchone()[0]

    print(f"Database contains {row_count:,} API users.")

    connection.close()

    print()
    print("API database load complete.")
    print(f"Table created: api_users")


if __name__ == "__main__":
    load_api_users()
