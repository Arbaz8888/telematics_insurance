import pandas as pd
import joblib
import numpy as np

# Load driver features
df = pd.read_csv("data/driver_features.csv")

# Load trained model
model = joblib.load("models/risk_model.pkl")

# Prepare features (must match training exactly!)
X = df.drop(columns=["driver_id", "gender"], errors="ignore")

# Predict risk classes (0=safe, 1=risky)
df["predicted_risk_class"] = model.predict(X)

# Predict risk probability (handle single-class case)
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X)
    if proba.shape[1] == 2:  # normal case
        df["predicted_risk_prob"] = proba[:, 1]
    else:  # degenerate case (only one class present)
        df["predicted_risk_prob"] = np.zeros(len(df))
        if model.classes_[0] == 1:  # if only risky class exists
            df["predicted_risk_prob"] = np.ones(len(df))

# Save updated driver scores
output_path = "data/driver_scores.csv"
df.to_csv(output_path, index=False)

print("✅ Driver scores generated and saved at:", output_path)
print(df[["driver_id", "predicted_risk_class", "predicted_risk_prob"]].head())
