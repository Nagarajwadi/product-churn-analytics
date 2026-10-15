import os
from unittest.mock import patch

from fastapi import HTTPException

# Keep tests independent of the local .env and cloud database.
os.environ["API_DATA_MODE"] = "mock"

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] == "mock"


def test_summary():
    response = client.get("/api/v1/summary")

    assert response.status_code == 200
    data = response.json()

    assert data["source"] == "mock_demo_data"
    assert data["total_customers"] == 5000
    assert data["churn_rate"] is None


def test_segments():
    response = client.get("/api/v1/segments")

    assert response.status_code == 200
    assert response.json()["segments"] == []


def test_model_performance():
    response = client.get("/api/v1/model-performance")

    assert response.status_code == 200
    data = response.json()

    assert data["model"] == "Random Forest"
    assert data["roc_auc"] == 0.953
    assert data["average_precision"] == 0.402


def test_product_analytics_not_published_in_mock_mode():
    response = client.get("/api/v1/product-analytics")

    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "mock_demo_data"
    assert data["metrics"] is None
    assert data["last_updated"] is None



def test_supabase_missing_environment_variables_returns_503():
    with (
        patch("api.main.API_DATA_MODE", "supabase"),
        patch("api.main.SUPABASE_URL", None),
        patch("api.main.SUPABASE_KEY", None),
    ):
        response = client.get("/api/v1/product-analytics")

    assert response.status_code == 503
    assert "environment variables are missing" in response.json()["detail"]


def test_supabase_client_initialization_failure_returns_503():
    with (
        patch("api.main.API_DATA_MODE", "supabase"),
        patch("api.main.SUPABASE_URL", "https://example.supabase.co"),
        patch("api.main.SUPABASE_KEY", "test-publishable-key"),
        patch(
            "api.main.create_client",
            side_effect=RuntimeError("connection initialization failed"),
        ),
    ):
        response = client.get("/api/v1/product-analytics")

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Unable to initialize the cloud data connection."
    )


def test_supabase_read_failure_returns_503():
    with (
        patch("api.main.API_DATA_MODE", "supabase"),
        patch(
            "api.main.latest_row",
            side_effect=HTTPException(
                status_code=503,
                detail="Unable to read approved metrics from "
                "product_analytics_metrics.",
            ),
        ),
    ):
        response = client.get("/api/v1/product-analytics")

    assert response.status_code == 503
    assert "Unable to read approved metrics" in response.json()["detail"]


def test_unsupported_data_mode_returns_500():
    with patch("api.main.API_DATA_MODE", "unsupported"):
        response = client.get("/api/v1/product-analytics")

    assert response.status_code == 500
    assert response.json()["detail"] == "Unsupported API_DATA_MODE."
