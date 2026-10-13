import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path("api/prepare_segments.py")
spec = importlib.util.spec_from_file_location("prepare_segments", MODULE_PATH)
prepare_segments = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_segments)


def test_summarize_calculates_aggregate_metrics():
    df = pd.DataFrame({"churn": [0, 1, 0, 1, 1, 0]})
    labels = pd.Series(["A", "A", "A", "B", "B", "B"])

    rows = prepare_segments.summarize(df, "test_type", labels)
    by_name = {row["segment_name"]: row for row in rows}

    assert by_name["A"]["total_customers"] == 3
    assert by_name["A"]["churned_customers"] == 1
    assert by_name["A"]["churn_rate"] == 0.3333

    assert by_name["B"]["total_customers"] == 3
    assert by_name["B"]["churned_customers"] == 2
    assert by_name["B"]["churn_rate"] == 0.6667


def test_sql_quote_escapes_single_quotes():
    assert prepare_segments.sql_quote("customer's group") == "'customer''s group'"


def test_preview_contains_no_customer_identifiers():
    preview = pd.read_json("api/generated_segments_preview.json")

    assert "user_id" not in preview.columns
    assert len(preview) == 9
    assert preview["total_customers"].sum() > 0
    assert set(preview["segment_type"]) == {
        "activity_frequency",
        "activity_recency",
        "purchase_rate",
    }
