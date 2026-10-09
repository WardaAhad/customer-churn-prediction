import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.index import FEATURES, app

client = TestClient(app)

CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80.0, "TotalCharges": 160.0,
}


def test_health():
    assert client.get("/api/health").json() == {"status": "ok"}


def test_home_page_is_served_locally():
    response = client.get("/")
    assert response.status_code == 200 and "Customer churn desk" in response.text


def test_feature_importance_has_ten_named_features():
    features = client.get("/api/feature-importance").json()["features"]
    assert len(features) == 10 and all(item["feature"] in FEATURES for item in features)


def test_risky_customer_scores_higher_than_loyal_customer():
    risky = client.post("/api/predict", json=CUSTOMER).json()
    loyal = client.post("/api/predict", json={**CUSTOMER, "tenure": 70, "Contract": "Two year",
                                              "InternetService": "DSL", "OnlineSecurity": "Yes",
                                              "TechSupport": "Yes", "MonthlyCharges": 55.0,
                                              "TotalCharges": 3850.0}).json()
    assert risky["probability"] > loyal["probability"]
    assert risky["risk_level"] in {"Medium", "High"} and loyal["risk_level"] == "Low"


def test_blank_total_charges_is_accepted():
    assert client.post("/api/predict", json={**CUSTOMER, "TotalCharges": None}).status_code == 200


@pytest.mark.parametrize("change", [{"Contract": "Weekly"}, {"tenure": -1}, {"MonthlyCharges": -5}, {"gender": ""}])
def test_invalid_single_customer_is_rejected(change):
    assert client.post("/api/predict", json={**CUSTOMER, **change}).status_code == 422


def test_batch_with_original_dataset_including_blank_total_charges():
    csv = open("data/Telco-Customer-Churn.csv", "rb").read()
    response = client.post("/api/predict-batch", content=csv, headers={"Content-Type": "text/csv"})
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["total_customers"] == 7043
    assert body["summary"]["predicted_churn"] + body["summary"]["retained"] == 7043


def test_batch_with_empty_extra_cells_is_json_safe():
    frame = pd.DataFrame([CUSTOMER]).assign(Notes=[None])
    response = client.post("/api/predict-batch", content=frame.to_csv(index=False))
    assert response.status_code == 200 and response.json()["results"][0]["Notes"] is None


def test_batch_missing_columns_and_bad_values():
    assert client.post("/api/predict-batch", content="a,b\n1,2\n").status_code == 422
    bad = pd.DataFrame([{**CUSTOMER, "Contract": "Nope"}]).to_csv(index=False)
    assert client.post("/api/predict-batch", content=bad).status_code == 422
    assert client.post("/api/predict-batch", content="").status_code == 400
