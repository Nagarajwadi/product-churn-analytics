import sqlite3


DATABASE_FILE = "data/product_analytics.db"


def create_enrichment_table():
    """Create a user-level table combining product events with API enrichment."""
    connection = sqlite3.connect(DATABASE_FILE)

    query = """
    CREATE TABLE IF NOT EXISTS user_api_enrichment AS
    SELECT
        e.user_id,
        COUNT(*) AS event_count,
        a.age,
        a.gender,
        a.role,
        a.department,
        a.job_title,
        a.country
    FROM clean_events e
    INNER JOIN api_users a
        ON CAST(SUBSTR(e.user_id, 2) AS INTEGER) = a.external_user_id
    GROUP BY
        e.user_id,
        a.age,
        a.gender,
        a.role,
        a.department,
        a.job_title,
        a.country
    """

    connection.execute("DROP TABLE IF EXISTS user_api_enrichment")
    connection.execute(query)

    row_count = connection.execute(
        "SELECT COUNT(*) FROM user_api_enrichment"
    ).fetchone()[0]

    connection.commit()
    connection.close()

    print("API ENRICHMENT COMPLETE")
    print(f"Enriched users: {row_count}")
    print("Table created: user_api_enrichment")


if __name__ == "__main__":
    create_enrichment_table()
