"""Run the complete API ingestion and enrichment workflow."""

from etl.api_ingestion import run_ingestion
from sql.load_api_users import load_api_users
from sql.create_api_enrichment import create_enrichment_table


def main():
    """Execute API ingestion, database loading, and enrichment."""
    print("=" * 50)
    print("API INGESTION AND ENRICHMENT PIPELINE")
    print("=" * 50)

    print("\n[1/3] Ingesting API data...")
    run_ingestion()

    print("\n[2/3] Loading API users into SQLite...")
    load_api_users()

    print("\n[3/3] Creating the enrichment table...")
    create_enrichment_table()

    print("\n" + "=" * 50)
    print("API PIPELINE COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()