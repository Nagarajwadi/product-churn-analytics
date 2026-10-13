"""Generate reviewed, aggregate-only churn segment SQL from the ML dataset."""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "ml_dataset.csv"
OUTPUT_PATH = ROOT / "api" / "generated_segments.sql"
PREVIEW_PATH = ROOT / "api" / "generated_segments_preview.json"

REQUIRED_COLUMNS = {
    "user_id",
    "active_days",
    "days_since_last_activity",
    "purchase_rate",
    "churn",
}


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def summarize(df: pd.DataFrame, segment_type: str, labels: pd.Series) -> list[dict]:
    working = pd.DataFrame(
        {
            "segment_name": labels,
            "churn": df["churn"],
        }
    ).dropna(subset=["segment_name"])

    grouped = (
        working.groupby("segment_name", observed=True)["churn"]
        .agg(total_customers="size", churned_customers="sum")
        .reset_index()
    )

    rows = []
    for row in grouped.to_dict("records"):
        total = int(row["total_customers"])
        churned = int(row["churned_customers"])
        rows.append(
            {
                "segment_type": segment_type,
                "segment_name": str(row["segment_name"]),
                "total_customers": total,
                "churned_customers": churned,
                "churn_rate": round(churned / total, 4) if total else 0.0,
            }
        )
    return rows


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {sorted(missing)}")

    if df["user_id"].duplicated().any():
        raise SystemExit("Expected one row per user; duplicate user_id values found.")

    if df["churn"].isna().any() or not set(df["churn"].unique()).issubset({0, 1}):
        raise SystemExit("Expected a non-null binary churn column containing only 0 and 1.")

    # Use quantile bands so highly skewed engagement values remain interpretable.
    active_labels = pd.qcut(
        df["active_days"].rank(method="first"),
        q=3,
        labels=["Low activity", "Medium activity", "High activity"],
    )

    recency_labels = pd.cut(
        df["days_since_last_activity"],
        bins=[-1, 24, 63, float("inf")],
        labels=["Recently active (0–24 days)", "Moderately inactive (25–63 days)", "Long inactive (64+ days)"],
    )

    purchase_labels = pd.qcut(
        df["purchase_rate"].rank(method="first"),
        q=3,
        labels=["Low purchase rate", "Medium purchase rate", "High purchase rate"],
    )

    rows = []
    rows.extend(summarize(df, "activity_frequency", active_labels))
    rows.extend(summarize(df, "activity_recency", recency_labels))
    rows.extend(summarize(df, "purchase_rate", purchase_labels))

    # Safety checks: aggregate counts must not exceed the source dataset.
    for segment_type in {r["segment_type"] for r in rows}:
        total = sum(r["total_customers"] for r in rows if r["segment_type"] == segment_type)
        churned = sum(r["churned_customers"] for r in rows if r["segment_type"] == segment_type)
        if total > len(df) or churned > int(df["churn"].sum()):
            raise SystemExit(f"Aggregate validation failed for {segment_type}.")

    # Generate SQL only; this script never connects to Supabase.
    statements = [
        "-- REVIEW BEFORE RUNNING: aggregate-only segment metrics.",
        "-- Generated from data/processed/ml_dataset.csv; no customer-level rows included.",
        "begin;",
        "delete from public.churn_segments where segment_type in "
        "('activity_frequency', 'activity_recency', 'purchase_rate');",
    ]

    for row in rows:
        statements.append(
            "insert into public.churn_segments "
            "(segment_type, segment_name, total_customers, churned_customers, churn_rate) values "
            f"({sql_quote(row['segment_type'])}, {sql_quote(row['segment_name'])}, "
            f"{row['total_customers']}, {row['churned_customers']}, {row['churn_rate']:.4f});"
        )

    statements.append("commit;")
    OUTPUT_PATH.write_text("\n".join(statements) + "\n")
    PREVIEW_PATH.write_text(json.dumps(rows, indent=2) + "\n")

    print(f"Source customers: {len(df):,}")
    print(f"Source churned customers: {int(df['churn'].sum()):,}")
    print(f"Aggregate segment rows generated: {len(rows)}")
    print(f"SQL for manual review: {OUTPUT_PATH.relative_to(ROOT)}")
    print(f"Aggregate preview: {PREVIEW_PATH.relative_to(ROOT)}")
    print("\nPreview:")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
