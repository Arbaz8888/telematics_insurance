"""
File: train_ai_models.py
Purpose: Train AI models (LSTM, Autoencoder, RandomForest baseline) on telematics training data.
         Save model weights, metrics, and ROC curves for dashboard integration.
Author: Arbaz Attar
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import joblib

# -------------------------------
# Configurations
# -------------------------------
DATA_DIR = "data"
MODEL_DIR = "models"

INPUT_FILE = os.path.join(DATA_DIR, "ai_training_data.csv")
RESULTS_FILE = os.path.join(DATA_DIR, "ai_model_results.csv")
ROC_FILE = os.path.join(DATA_DIR, "ai_roc_curves.csv")

os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------------
# Utility: Save Metrics
# -------------------------------
def save_metrics(model_name, y_true, y_pred, y_proba, results, roc_curves):
    """Safely compute and append metrics + ROC curve."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

    # Handle NaN/Inf in probabilities
    y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=1.0, neginf=0.0)

    # Skip ROC if predictions are constant
    if np.all(y_proba == y_proba[0]):
        print(f"⚠ {model_name} produced constant predictions, skipping ROC.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    for i in range(len(fpr)):
        roc_curves.append({
            "model": model_name,
            "fpr": fpr[i],
            "tpr": tpr[i]
        })


# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv(INPUT_FILE)

X = df[["avg_speed", "max_speed", "braking_events", "cornering_events", "trip_duration"]]
y = df["trip_risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results, roc_curves = [], []


# -------------------------------
# Model 1: RandomForest (baseline)
# -------------------------------
print("Training RandomForest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

save_metrics("RandomForest", y_test, y_pred, y_proba, results, roc_curves)
joblib.dump(rf, os.path.join(MODEL_DIR, "randomforest.pkl"))
print("✅ RandomForest saved.")


# -------------------------------
# Model 2: LSTM (sequence-aware AI model)
# -------------------------------
print("Training LSTM...")
X_train_lstm = np.expand_dims(X_train.values, axis=1)
X_test_lstm = np.expand_dims(X_test.values, axis=1)

lstm = Sequential()
lstm.add(Input(shape=(1, X_train.shape[1])))
lstm.add(LSTM(32))
lstm.add(Dropout(0.2))
lstm.add(Dense(1, activation="sigmoid"))

lstm.compile(optimizer=Adam(learning_rate=0.001),
             loss="binary_crossentropy", metrics=["accuracy"])
lstm.fit(X_train_lstm, y_train, epochs=30, batch_size=8,
         validation_split=0.2, verbose=0,
         callbacks=[EarlyStopping(monitor="val_loss", patience=5)])

y_pred_proba = lstm.predict(X_test_lstm).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)

save_metrics("LSTM", y_test, y_pred, y_pred_proba, results, roc_curves)
lstm.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
print("✅ LSTM saved.")


# -------------------------------
# Model 3: Autoencoder (unsupervised anomaly detection)
# -------------------------------
print("Training Autoencoder...")
X_mean, X_std = X_train.mean(), X_train.std()
X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

input_dim = X_train.shape[1]
inp = Input(shape=(input_dim,))
encoded = Dense(16, activation="relu")(inp)
encoded = Dense(8, activation="relu")(encoded)
decoded = Dense(16, activation="relu")(encoded)
out = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inp, out)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train_norm, X_train_norm, epochs=50, batch_size=8,
                validation_split=0.2, verbose=0,
                callbacks=[EarlyStopping(monitor="val_loss", patience=5)])

X_test_pred = autoencoder.predict(X_test_norm)
mse = np.mean(np.square(X_test_norm - X_test_pred), axis=1)

# Clean NaNs
mse = np.nan_to_num(mse, nan=0.0, posinf=1.0, neginf=0.0)

threshold = np.percentile(mse, 80)  # top 20% risky
y_pred = (mse > threshold).astype(int)

save_metrics("Autoencoder", y_test, y_pred, mse, results, roc_curves)
autoencoder.save(os.path.join(MODEL_DIR, "autoencoder_model.keras"))
print("✅ Autoencoder saved.")


# -------------------------------
# Save Results
# -------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(RESULTS_FILE, index=False)

roc_df = pd.DataFrame(roc_curves)
roc_df.to_csv(ROC_FILE, index=False)

print(f"✅ Training complete. Results saved to {RESULTS_FILE} and {ROC_FILE}")
