"""Tests for the FastAPI service."""

import pytest
from fastapi.testclient import TestClient

from api import app
from tests.test_model import APPLICANT


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_docs_endpoint_is_available(client):
    """Used as the deploy health check, so it must stay reachable."""
    assert client.get("/docs").status_code == 200


def test_predict_success(client):
    response = client.post("/predict", json=APPLICANT)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["data"]["approve"] in (0, 1)


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in APPLICANT.items() if k != "cibil_score"}

    assert client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_wrong_type(client):
    payload = {**APPLICANT, "cibil_score": "not-a-number"}

    assert client.post("/predict", json=payload).status_code == 422
