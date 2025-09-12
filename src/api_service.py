# src/api_service.py
from fastapi import FastAPI, HTTPException, Header
import pandas as pd
import joblib

app = FastAPI(title="Telematics Insurance API", version="1.0")

# Load models
try:
    rf_model = joblib.load("models/risk_model_randomforest.pkl")
except:
    rf_model = None

# Load data
try:
    driver_features = pd.read_csv("data/driver_features.csv")
    policies = pd.read_csv("data/policies.csv")
    scores = pd.read_csv("data/driver_scores.csv")
except:
    driver_features, policies, scores = None, None, None


# Security: simple API key
API_KEY = "INSURITY_DEMO_KEY"


def verify_api_key(key: str):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


@app.get("/health")
def health():
    return {"status": "ok", "message": "Telematics Insurance API running"}


@app.get("/predict/{driver_id}")
def predict_driver_risk(driver_id: str, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)

    if driver_features is None or rf_model is None:
        raise HTTPException(status_code=500, detail="Models or data not available")

    row = driver_features[driver_features["driver_id"] == driver_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="Driver not found")

    X = row.drop(columns=["driver_id", "gender"], errors="ignore")
    prob = float(rf_model.predict_proba(X)[0][1])

    return {
        "driver_id": driver_id,
        "predicted_risk_prob": prob,
        "message": "Risk score generated successfully"
    }


@app.get("/premium/{driver_id}")
def get_adjusted_premium(driver_id: str, x_api_key: str = Header(...)):
    verify_api_key(x_api_key)

    if scores is None or policies is None:
        raise HTTPException(status_code=500, detail="Policies or scores not available")

    policy = policies[policies["driver_id"] == driver_id]
    score = scores[scores["driver_id"] == driver_id]

    if policy.empty or score.empty:
        raise HTTPException(status_code=404, detail="Driver policy or score not found")

    base = float(policy.iloc[0]["premium_amount"])
    risk_prob = float(score.iloc[0]["predicted_risk_prob"])
    adjusted = round(base * (1 + 0.5 * risk_prob), 2)

    return {
        "driver_id": driver_id,
        "base_premium": base,
        "risk_prob": risk_prob,
        "adjusted_premium": adjusted,
        "currency": "USD"
    }
