# src/train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# -------------------------------
# Label generation from claims
# -------------------------------
def generate_labels_from_claims(features_path="data/driver_features_decrypted.csv",
                                history_path="data/vehicle_history.csv"):
    """
    Generate high-risk labels for drivers using claims data.
    Risky = drivers with multiple claims or older vehicles.
    """
    features = pd.read_csv(features_path)
    history = pd.read_csv(history_path)

    df = features.merge(history, on="driver_id", how="left")

    df["label_high_risk"] = (
        (df["claims_count"] > 2) |
        ((df["vehicle_age"] > 10) & (df["claims_count"] > 0)) |
        ((df["claims_count"] > 0) &
         ((df.get("count_HARD_BRAKE", 0) > df["total_trips"]) |
          (df.get("count_SPEEDING", 0) > df["total_trips"] * 0.5)))
    ).astype(int)

    # Add light noise for realism
    rng = np.random.default_rng(42)
    flip_idx = df.sample(frac=0.05, random_state=42).index
    df.loc[flip_idx, "label_high_risk"] = 1 - df.loc[flip_idx, "label_high_risk"]

    print("Label distribution (0=safe, 1=risky):")
    print(df["label_high_risk"].value_counts())
    return df


# -------------------------------
# Load Data & Labels
# -------------------------------
df = generate_labels_from_claims(
    features_path="data/driver_features_decrypted.csv",
    history_path="data/vehicle_history.csv"
)

# Features & Target
drop_cols = [c for c in ["driver_id", "anon_id", "gender", "label_high_risk"] if c in df.columns]
X = df.drop(columns=drop_cols, errors="ignore")
y = df["label_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------
# Train Random Forest
# -------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save model
# -------------------------------
os.makedirs("models", exist_ok=True)
model_path = "models/risk_model.pkl"
joblib.dump(clf, model_path)
print(f"\n✅ Model saved at {model_path}")
