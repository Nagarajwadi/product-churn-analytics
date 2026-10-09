import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.api_ingestion import validate_users


def test_validate_users_rejects_empty_list():
    try:
        validate_users([])
    except ValueError as error:
        assert str(error) == "API returned no user records"
    else:
        raise AssertionError("Expected ValueError for empty user list")


def test_validate_users_rejects_duplicate_ids():
    users = [
        {
            "external_user_id": 1,
            "age": 29,
            "gender": "female",
            "role": "admin",
            "department": "Engineering",
            "job_title": "Engineer",
            "country": "United States",
        },
        {
            "external_user_id": 1,
            "age": 36,
            "gender": "male",
            "role": "user",
            "department": "Support",
            "job_title": "Support Specialist",
            "country": "United States",
        },
    ]

    try:
        validate_users(users)
    except ValueError as error:
        assert str(error) == "Duplicate external_user_id values found"
    else:
        raise AssertionError("Expected ValueError for duplicate user IDs")


def test_validate_users_rejects_missing_fields():
    users = [
        {
            "external_user_id": 1,
            "age": 29,
            "gender": "female",
            "role": "admin",
            "department": "Engineering",
            "country": "United States",
        }
    ]

    try:
        validate_users(users)
    except ValueError as error:
        assert "missing fields" in str(error)
        assert "job_title" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing required fields")


def test_validate_users_rejects_non_dictionary_record():
    users = ["invalid user record"]

    try:
        validate_users(users)
    except ValueError as error:
        assert str(error) == "Each user record must be a dictionary"
    else:
        raise AssertionError("Expected ValueError for non-dictionary record")


def test_validate_users_rejects_missing_external_user_id():
    users = [
        {
            "age": 29,
            "gender": "female",
            "role": "admin",
            "department": "Engineering",
            "job_title": "Engineer",
            "country": "United States",
        }
    ]

    try:
        validate_users(users)
    except ValueError as error:
        assert "external_user_id" in str(error)
    else:
        raise AssertionError("Expected ValueError for missing external user ID")
