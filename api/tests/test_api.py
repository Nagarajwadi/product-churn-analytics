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