import math
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

load_dotenv()

app = FastAPI(
    title="Product Churn Analytics API",
    description="Read-only API for approved aggregate churn analytics.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://productanalytics.me",
        "https://www.productanalytics.me",
    ],
    allow_origin_regex=r"https://[a-z0-9-]+-8000\.app\.github\.dev",
    allow_credentials=False,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

API_DATA_MODE = os.getenv("API_DATA_MODE", "mock").strip().lower()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(
            status_code=503,
            detail="Supabase is selected but its environment variables are missing.",
        )

    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Unable to initialize the cloud data connection.",
        ) from exc


def latest_row(table_name: str):
    try:
        response = (
            get_supabase_client()
            .table(table_name)
            .select("*")
            .order("published_at", desc=True)
            .limit(1)
            .execute()
        )
        return response.data[0] if response.data else None
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to read approved metrics from {table_name}.",
        ) from exc


def all_rows(table_name: str):
    try:
        response = (
            get_supabase_client()
            .table(table_name)
            .select("*")
            .order("published_at", desc=True)
            .execute()
        )
        return response.data
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to read approved metrics from {table_name}.",
        ) from exc


@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "service": "product-churn-analytics-api",
        "mode": API_DATA_MODE,
    }


@app.get("/api/v1/product-analytics", tags=["Product Analytics"])
def get_product_analytics():
    if API_DATA_MODE == "mock":
        return {
            "source": "mock_demo_data",
            "metrics": None,
            "last_updated": None,
            "note": "No Product Analytics snapshot has been published.",
        }

    if API_DATA_MODE != "supabase":
        raise HTTPException(status_code=500, detail="Unsupported API_DATA_MODE.")

    row = latest_row("product_analytics_metrics")
    if row is None:
        return {
            "source": "supabase",
            "metrics": None,
            "last_updated": None,
            "note": "No Product Analytics snapshot has been published yet.",
        }

    fields = (
        "total_customers", "total_events", "total_sessions",
        "total_logins", "total_product_views", "avg_active_days",
        "avg_search_rate", "avg_cart_rate", "avg_purchase_rate",
        "avg_days_since_last_activity", "avg_days_since_last_login",
        "avg_days_since_last_product_view", "avg_events_per_active_day",
        "avg_sessions_per_active_day",
    )
    try:
        if not isinstance(row, dict):
            raise ValueError("Snapshot must be an object.")

        metrics = {}

        for field in fields:
            value = row[field]

            if field.startswith("avg_"):
                number = float(value)

                if not math.isfinite(number) or number < 0:
                    raise ValueError("Average metric must be finite and non-negative.")

                if field in {
                    "avg_search_rate",
                    "avg_cart_rate",
                    "avg_purchase_rate",
                } and number > 1:
                    raise ValueError("Rate must be between 0 and 1.")

                metrics[field] = number
            else:
                if isinstance(value, bool):
                    raise ValueError("Count must be an integer.")

                if isinstance(value, int):
                    number = value
                elif isinstance(value, str) and value.strip().isdecimal():
                    number = int(value.strip())
                else:
                    raise ValueError("Count must be an integer.")

                if number < 0:
                    raise ValueError("Count must be non-negative.")

                metrics[field] = number

        published_at = row["published_at"]

        if not isinstance(published_at, str) or not published_at.strip():
            raise ValueError("Snapshot timestamp is missing.")

        datetime.fromisoformat(published_at.strip().replace("Z", "+00:00"))

    except (KeyError, TypeError, ValueError, OverflowError):
        raise HTTPException(
            status_code=500,
            detail="Published Product Analytics snapshot is invalid.",
        ) from None


    return {
        "source": "supabase",
        "metrics": metrics,
        "last_updated": published_at,
    }


@app.get("/api/v1/summary", tags=["Analytics"])
def get_summary():
    if API_DATA_MODE == "mock":
        return {
            "source": "mock_demo_data",
            "total_customers": 5000,
            "churned_customers": None,
            "churn_rate": None,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "note": "Demo placeholder; verified churn metrics have not been published.",
        }

    if API_DATA_MODE != "supabase":
        raise HTTPException(status_code=500, detail="Unsupported API_DATA_MODE.")

    row = latest_row("churn_summary")
    if row is None:
        return {
            "source": "supabase",
            "total_customers": None,
            "churned_customers": None,
            "churn_rate": None,
            "last_updated": None,
            "note": "No summary metrics have been published yet.",
        }

    return {
        "source": "supabase",
        "total_customers": row["total_customers"],
        "churned_customers": row["churned_customers"],
        "churn_rate": float(row["churn_rate"]),
        "last_updated": row["published_at"],
    }


@app.get("/api/v1/segments", tags=["Analytics"])
def get_segments():
    if API_DATA_MODE == "mock":
        return {
            "source": "mock_demo_data",
            "segments": [],
            "note": "No segment data has been published yet.",
        }

    if API_DATA_MODE != "supabase":
        raise HTTPException(status_code=500, detail="Unsupported API_DATA_MODE.")

    return {
        "source": "supabase",
        "segments": all_rows("churn_segments"),
    }


@app.get("/api/v1/model-performance", tags=["Machine Learning"])
def get_model_performance():
    if API_DATA_MODE == "mock":
        return {
            "source": "verified_project_metrics",
            "model": "Random Forest",
            "roc_auc": 0.953,
            "average_precision": 0.402,
            "evaluation": "held-out test set",
        }

    if API_DATA_MODE != "supabase":
        raise HTTPException(status_code=500, detail="Unsupported API_DATA_MODE.")

    row = latest_row("model_metrics")
    if row is None:
        return {
            "source": "supabase",
            "model": None,
            "roc_auc": None,
            "average_precision": None,
            "evaluation": None,
            "note": "No model metrics have been published yet.",
        }

    return {
        "source": "supabase",
        "model": row["model_name"],
        "roc_auc": float(row["roc_auc"]),
        "average_precision": float(row["average_precision"]),
        "evaluation": row["evaluation_method"],
        "last_updated": row["published_at"],
    }
