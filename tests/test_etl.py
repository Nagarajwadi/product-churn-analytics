import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "etl"))

from etl_pipeline import transform_data, validate_data


def test_transform_removes_duplicates():
    data = pd.DataFrame(
        {
            "user_id": ["U00001", "U00001", "U00002"],
            "timestamp": [
                "2026-01-02 10:00:00",
                "2026-01-02 10:00:00",
                "2026-01-01 09:00:00",
            ],
            "event_name": ["login", "login", "signup"],
        }
    )

    result = transform_data(data)

    assert len(result) == 2


def test_transform_sorts_by_user_and_timestamp():
    data = pd.DataFrame(
        {
            "user_id": ["U00002", "U00001", "U00001"],
            "timestamp": [
                "2026-01-02 10:00:00",
                "2026-01-03 10:00:00",
                "2026-01-01 09:00:00",
            ],
            "event_name": ["login", "login", "signup"],
        }
    )

    result = transform_data(data)

    assert list(result["user_id"]) == ["U00001", "U00001", "U00002"]
    assert list(result["timestamp"]) == [
        pd.Timestamp("2026-01-01 09:00:00"),
        pd.Timestamp("2026-01-03 10:00:00"),
        pd.Timestamp("2026-01-02 10:00:00"),
    ]

    from etl_pipeline import validate_data


def test_validate_accepts_valid_events():
    data = pd.DataFrame(
        {
            "user_id": ["U00001"],
            "timestamp": [pd.Timestamp("2026-01-01 10:00:00")],
            "event_name": ["login"],
            "device": ["web"],
            "country": ["India"],
            "plan": ["free"],
            "session_id": ["U00001_S0001"],
        }
    )

    assert validate_data(data) is True


def test_validate_rejects_missing_columns():
    data = pd.DataFrame(
        {
            "user_id": ["U00001"],
            "timestamp": [pd.Timestamp("2026-01-01 10:00:00")],
            "event_name": ["login"],
        }
    )

    try:
        validate_data(data)
        assert False, "Expected ValueError for missing columns"
    except ValueError as error:
        assert "Missing columns" in str(error)


def test_validate_rejects_invalid_event():
    data = pd.DataFrame(
        {
            "user_id": ["U00001"],
            "timestamp": [pd.Timestamp("2026-01-01 10:00:00")],
            "event_name": ["invalid_event"],
            "device": ["web"],
            "country": ["India"],
            "plan": ["free"],
            "session_id": ["U00001_S0001"],
        }
    )

    try:
        validate_data(data)
        assert False, "Expected ValueError for invalid event"
    except ValueError as error:
        assert "Invalid event types" in str(error)