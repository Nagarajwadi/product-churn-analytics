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
