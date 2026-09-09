from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "Explainable Heart Disease" in body["name"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_invalid_payload_rejected():
    response = client.post("/predict", json={"age": 55})
    assert response.status_code == 422
