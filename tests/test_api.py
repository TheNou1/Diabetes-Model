"""Example-driven tests for the Diabetes Predictor API.

Run from the project root with:
    pytest tests/test_api.py -v

Uses FastAPI's TestClient, so no server needs to be running first.
Requires a trained model to already exist at models/diabetes_model.pkl
(run `python -m src.train` if this is missing).
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------
# Example inputs. Each is a realistic row of the 10 sklearn diabetes
# features (already mean-centered/scaled by sklearn, hence small floats).
# ---------------------------------------------------------------------
EXAMPLE_PATIENTS = [
    {
        "name": "baseline / near-average patient",
        "payload": {
            "age": 0.0, "sex": 0.0, "bmi": 0.0, "bp": 0.0,
            "s1": 0.0, "s2": 0.0, "s3": 0.0, "s4": 0.0,
            "s5": 0.0, "s6": 0.0,
        },
    },
    {
        "name": "higher bmi and blood pressure",
        "payload": {
            "age": 0.03, "sex": 0.05, "bmi": 0.06, "bp": 0.02,
            "s1": -0.04, "s2": -0.03, "s3": -0.02, "s4": 0.01,
            "s5": 0.02, "s6": 0.01,
        },
    },
    {
        "name": "lower-than-average values across the board",
        "payload": {
            "age": -0.05, "sex": -0.04, "bmi": -0.06, "bp": -0.03,
            "s1": 0.02, "s2": 0.01, "s3": 0.03, "s4": -0.02,
            "s5": -0.01, "s6": -0.02,
        },
    },
    {
        "name": "extreme / boundary values",
        "payload": {
            "age": 0.11, "sex": 0.05, "bmi": 0.17, "bp": 0.13,
            "s1": 0.15, "s2": 0.16, "s3": -0.10, "s4": 0.18,
            "s5": 0.13, "s6": 0.14,
        },
    },
]


def test_root_endpoint_is_healthy():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize(
    "case", EXAMPLE_PATIENTS, ids=[c["name"] for c in EXAMPLE_PATIENTS]
)
def test_predict_returns_a_number(case):
    response = client.post("/predict", json=case["payload"])
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)

    # Sanity range check: diabetes progression score in this dataset
    # (per sklearn docs) roughly spans 25 to 350. A wildly out-of-range
    # prediction would suggest something is wrong with the model or
    # feature order, not just an unlucky prediction.
    assert 0 <= body["prediction"] <= 400


def test_predict_rejects_missing_field():
    incomplete_payload = {
        "age": 0.03, "sex": 0.05, "bmi": 0.06, "bp": 0.02,
        "s1": -0.04, "s2": -0.03, "s3": -0.02, "s4": 0.01,
        "s5": 0.02,
        # "s6" missing on purpose
    }
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422  # FastAPI validation error


def test_predict_rejects_wrong_type():
    bad_payload = {
        "age": "not-a-number", "sex": 0.05, "bmi": 0.06, "bp": 0.02,
        "s1": -0.04, "s2": -0.03, "s3": -0.02, "s4": 0.01,
        "s5": 0.02, "s6": 0.01,
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
