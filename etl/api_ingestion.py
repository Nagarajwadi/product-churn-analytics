import json
from pathlib import Path

from src.api_client import fetch_all_users, transform_users


OUTPUT_PATH = Path("data/raw/api_users.json")


def validate_users(users):
    """Validate the API records before saving."""
    if not users:
        raise ValueError("API returned no user records")

    required_fields = {
        "external_user_id",
        "age",
        "gender",
        "role",
        "department",
        "job_title",
        "country",
    }

    for user in users:
        if not isinstance(user, dict):
            raise ValueError("Each user record must be a dictionary")

        missing_fields = required_fields - user.keys()

        if missing_fields:
            raise ValueError(
                f"User {user.get('external_user_id')} "
                f"is missing fields: {missing_fields}"
            )

    for user in users:
        external_user_id = user["external_user_id"]

        if external_user_id is None:
            raise ValueError("external_user_id cannot be None")

    if len(users) != len({user["external_user_id"] for user in users}):
        raise ValueError("Duplicate external_user_id values found")


def run_ingestion():
    """Fetch, transform, validate, and save API data."""
    raw_users = fetch_all_users(page_size=50)
    users = transform_users(raw_users)

    validate_users(users)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(users, file, indent=2)

    print("API INGESTION COMPLETE")
    print(f"Records fetched: {len(users)}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_ingestion()