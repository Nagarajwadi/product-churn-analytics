"""Generate a reviewable aggregate-only Product Analytics SQL snapshot."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "ml_dataset.csv"
SQL_OUT = ROOT / "api" / "generated_product_analytics.sql"
PREVIEW_OUT = ROOT / "api" / "generated_product_analytics_preview.json"

COUNTS = [
    "total_events", "total_sessions", "login_count", "product_view_count"
]
AVERAGES = [
    "active_days", "search_rate", "cart_rate", "purchase_rate",
    "days_since_last_activity", "days_since_last_login",
    "days_since_last_product_view", "events_per_active_day",
    "sessions_per_active_day",
]
REQUIRED = {"user_id", "churn", *COUNTS, *AVERAGES}


def main():
    df = pd.read_csv(DATA)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise SystemExit("Dataset is empty.")
    if df["user_id"].isna().any() or df["user_id"].duplicated().any():
        raise SystemExit("User IDs must be present and unique.")
    if df[list(REQUIRED - {"user_id"})].isna().any().any():
        raise SystemExit("Required metric columns contain missing values.")
    if not set(df["churn"].unique()).issubset({0, 1}):
        raise SystemExit("Unexpected churn values.")

    metrics = {
        "total_customers": int(len(df)),
        "total_events": int(df["total_events"].sum()),
        "total_sessions": int(df["total_sessions"].sum()),
        "total_logins": int(df["login_count"].sum()),
        "total_product_views": int(df["product_view_count"].sum()),
        "avg_active_days": round(float(df["active_days"].mean()), 4),
        "avg_search_rate": round(float(df["search_rate"].mean()), 6),
        "avg_cart_rate": round(float(df["cart_rate"].mean()), 6),
        "avg_purchase_rate": round(float(df["purchase_rate"].mean()), 6),
        "avg_days_since_last_activity": round(
            float(df["days_since_last_activity"].mean()), 4
        ),
        "avg_days_since_last_login": round(
            float(df["days_since_last_login"].mean()), 4
        ),
        "avg_days_since_last_product_view": round(
            float(df["days_since_last_product_view"].mean()), 4
        ),
        "avg_events_per_active_day": round(
            float(df["events_per_active_day"].mean()), 4
        ),
        "avg_sessions_per_active_day": round(
            float(df["sessions_per_active_day"].mean()), 4
        ),
    }

    columns = ", ".join(metrics)
    values = ", ".join(str(value) for value in metrics.values())
    sql = f"""-- REVIEW BEFORE RUNNING.
-- Aggregate snapshot generated from data/processed/ml_dataset.csv.
-- No customer-level rows or identifiers are included.
insert into public.product_analytics_metrics ({columns})
values ({values});
"""
    SQL_OUT.write_text(sql, encoding="utf-8")
    PREVIEW_OUT.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print("Generated:", SQL_OUT.relative_to(ROOT))
    print("Preview:", PREVIEW_OUT.relative_to(ROOT))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
