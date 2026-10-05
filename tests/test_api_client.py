import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api_client import transform_users


def test_transform_users_keeps_expected_fields():
    users = [
        {
            "id": 1,
            "age": 29,
            "gender": "female",
            "role": "admin",
            "company": {
                "department": "Engineering",
                "title": "Sales Manager",
            },
            "address": {
                "country": "United States",
            },
            "password": "should_not_be_kept",
            "ssn": "should_not_be_kept",
        }
    ]

    result = transform_users(users)

    assert result == [
        {
            "external_user_id": 1,
            "age": 29,
            "gender": "female",
            "role": "admin",
            "department": "Engineering",
            "job_title": "Sales Manager",
            "country": "United States",
        }
    ]


def test_transform_users_handles_multiple_users():
    users = [
        {
            "id": 1,
            "age": 29,
            "gender": "female",
            "role": "admin",
            "company": {
                "department": "Engineering",
                "title": "Sales Manager",
            },
            "address": {
                "country": "United States",
            },
        },
        {
            "id": 2,
            "age": 36,
            "gender": "male",
            "role": "user",
            "company": {
                "department": "Support",
                "title": "Support Specialist",
            },
            "address": {
                "country": "United States",
            },
        },
    ]

    result = transform_users(users)

    assert len(result) == 2
    assert result[0]["external_user_id"] == 1
    assert result[1]["external_user_id"] == 2
