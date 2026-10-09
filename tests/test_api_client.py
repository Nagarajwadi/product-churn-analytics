import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unittest.mock import Mock, patch

import requests

from src.api_client import fetch_all_users, fetch_users, transform_users


def test_fetch_all_users_paginates_until_total():
    pages = [
        {"users": [{"id": 1}, {"id": 2}], "total": 3},
        {"users": [{"id": 3}], "total": 3},
    ]

    with patch("src.api_client.fetch_users", side_effect=pages) as mock_fetch:
        result = fetch_all_users(page_size=2)

    assert result == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert mock_fetch.call_args_list[0].kwargs == {"limit": 2, "skip": 0}
    assert mock_fetch.call_args_list[1].kwargs == {"limit": 2, "skip": 2}


def test_fetch_all_users_stops_on_empty_page():
    pages = [
        {"users": [{"id": 1}], "total": 5},
        {"users": [], "total": 5},
    ]

    with patch("src.api_client.fetch_users", side_effect=pages) as mock_fetch:
        result = fetch_all_users(page_size=1)

    assert result == [{"id": 1}]
    assert mock_fetch.call_count == 2

def test_fetch_users_sends_pagination_parameters():
    response = Mock()
    response.json.return_value = {"users": [{"id": 1}], "total": 1}
    response.raise_for_status.return_value = None

    with patch("src.api_client.requests.get", return_value=response) as mock_get:
        result = fetch_users(limit=25, skip=50)

    assert result == {"users": [{"id": 1}], "total": 1}
    mock_get.assert_called_once_with(
        "https://dummyjson.com/users",
        params={"limit": 25, "skip": 50},
        headers={
            "User-Agent": "product-churn-analytics/1.0",
            "Accept": "application/json",
        },
        timeout=10,
    )


def test_fetch_users_rejects_invalid_response():
    response = Mock()
    response.json.return_value = {"total": 1}
    response.raise_for_status.return_value = None

    with patch("src.api_client.requests.get", return_value=response):
        try:
            fetch_users()
        except ValueError as exc:
            assert str(exc) == "API response does not contain 'users'"
        else:
            raise AssertionError("fetch_users() should reject an invalid response")


def test_fetch_users_propagates_http_errors():
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError("503 Service Unavailable")

    with patch("src.api_client.requests.get", return_value=response):
        try:
            fetch_users()
        except requests.HTTPError as exc:
            assert str(exc) == "503 Service Unavailable"
        else:
            raise AssertionError("fetch_users() should propagate HTTP errors")


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



def test_fetch_users_rejects_non_positive_limit():
    with patch("src.api_client.requests.get") as mock_get:
        try:
            fetch_users(limit=0)
        except ValueError as error:
            assert str(error) == "limit must be a positive integer"
        else:
            raise AssertionError("Expected ValueError for limit=0")

        mock_get.assert_not_called()


def test_fetch_users_rejects_negative_skip():
    with patch("src.api_client.requests.get") as mock_get:
        try:
            fetch_users(skip=-1)
        except ValueError as error:
            assert str(error) == "skip must be a non-negative integer"
        else:
            raise AssertionError("Expected ValueError for skip=-1")

        mock_get.assert_not_called()


def test_fetch_users_rejects_boolean_limit():
    with patch("src.api_client.requests.get") as mock_get:
        try:
            fetch_users(limit=True)
        except ValueError as error:
            assert str(error) == "limit must be a positive integer"
        else:
            raise AssertionError("Expected ValueError for limit=True")

        mock_get.assert_not_called()