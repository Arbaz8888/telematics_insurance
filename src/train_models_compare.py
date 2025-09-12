# src/train_models_compare.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve
)
import xgboost as xgb
import joblib
import os

# -------------------------------
# Label generation from claims
# -------------------------------
def generate_labels_from_claims(features_path="data/driver_features.csv",
                                history_path="data/vehicle_history.csv"):
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
    flip_idx = df.sample(frac=0.05, random_state=42).index
    df.loc[flip_idx, "label_high_risk"] = 1 - df.loc[flip_idx, "label_high_risk"]

    print("Label distribution (0=safe, 1=risky):")
    print(df["label_high_risk"].value_counts())
    return df


# -------------------------------
# Load Data & Labels
# -------------------------------
df = generate_labels_from_claims(
    features_path="data/driver_features.csv",
    history_path="data/vehicle_history.csv"
)

X = df.drop(columns=["driver_id", "gender", "label_high_risk"], errors="ignore")
y = df["label_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# -------------------------------
# Train multiple models
# -------------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "XGBoost": xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),
    "NeuralNetwork": MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )
}

results = []
roc_data = []

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
        y_prob = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f"models/risk_model_{name.lower()}.pkl")
    results.append([name, acc, f1, prec, rec, roc_auc])

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr})
    roc_data.append(roc_df)

# -------------------------------
# Save outputs
# -------------------------------
results_df = pd.DataFrame(results, columns=["model", "accuracy", "f1_score", "precision", "recall", "roc_auc"])
results_df.to_csv("data/model_comparison.csv", index=False)
print("\n✅ Model comparison saved at data/model_comparison.csv")

roc_all = pd.concat(roc_data, ignore_index=True)
roc_all.to_csv("data/roc_curves.csv", index=False)
print("✅ ROC curves saved at data/roc_curves.csv")

# Feature Importances
importances = pd.DataFrame({
    "feature": X.columns,
    "importance": models["RandomForest"].feature_importances_
}).sort_values("importance", ascending=False)
importances.to_csv("data/feature_importances.csv", index=False)
print("✅ Feature importances saved at data/feature_importances.csv")
