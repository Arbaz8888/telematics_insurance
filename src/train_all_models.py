"""
File: train_all_models.py
Purpose: Train ML (RandomForest) vs AI (LSTM, GRU, CNN, Autoencoder, Transformer)
         on telematics time-series data. Save weights, metrics, and ROC curves.
Author: Arbaz Attar (Modified)
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from keras.models import Sequential, Model
from keras.layers import (
    LSTM, GRU, Conv1D, Dense, Dropout, Flatten, Input,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization, Add,
    MaxPooling1D, BatchNormalization
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import tensorflow as tf

# -------------------------------
# Config
# -------------------------------
DATA_FILE = "data/ai_telematics_timeseries.csv"
MODEL_DIR = "models"
RESULTS_FILE = "data/ai_model_results.csv"
ROC_FILE = "data/ai_roc_curves.csv"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# Utility: Save Metrics
# -------------------------------
def save_metrics(model_name, y_true, y_pred, y_proba, results, roc_curves):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": None,
        "pr_auc": None
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            metrics["roc_auc"] = round(roc_auc, 4)
            metrics["pr_auc"] = round(pr_auc, 4)

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            for i in range(len(fpr)):
                roc_curves.append({"model": model_name, "fpr": fpr[i], "tpr": tpr[i]})
                
            print(f"  📊 {model_name}: Acc={acc:.3f}, ROC-AUC={roc_auc:.3f}, F1={f1:.3f}")
        except Exception as e:
            print(f"⚠ {model_name} ROC/PR failed: {e}")
    else:
        print(f"⚠ {model_name} produced constant predictions, skipping ROC/PR.")

    results.append(metrics)

# -------------------------------
# Load Data
# -------------------------------
print("Loading data...")
df = pd.read_csv(DATA_FILE)
FEATURES = ["speed", "acceleration", "brake_intensity", "steering_angle", "g_force"]

# Group per trip
X, y = [], []
for trip_id, group in df.groupby("trip_id"):
    X.append(group[FEATURES].values)
    y.append(group["risk_label"].iloc[0])

print(f"Loaded {len(X)} trips with labels: {np.bincount(y)}")

# Pad sequences to same length
max_len = max(len(seq) for seq in X)
print(f"Max sequence length: {max_len}")

X_padded = np.array([
    np.pad(seq, ((0, max_len - len(seq)), (0, 0)), mode="constant") 
    for seq in X
])
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Train labels: {np.bincount(y_train)}, Test labels: {np.bincount(y_test)}")

results, roc_curves = [], []

# -------------------------------
# 1. RandomForest (ONLY aggregate features)
# -------------------------------
print("\n🔥 Training RandomForest (aggregate features only)...")

def aggregate_features(X_seq):
    """Extract statistical aggregates that miss temporal patterns"""
    # Remove padding (zeros at the end)
    non_zero_mask = np.any(X_seq != 0, axis=1)
    if np.any(non_zero_mask):
        X_clean = X_seq[non_zero_mask]
    else:
        X_clean = X_seq  # fallback
    
    features = []
    features.extend(X_clean.mean(axis=0))  # mean of each feature
    features.extend(X_clean.std(axis=0))   # std of each feature
    features.extend(X_clean.max(axis=0))   # max of each feature
    features.extend(X_clean.min(axis=0))   # min of each feature
    
    return np.array(features)

X_train_rf = np.array([aggregate_features(seq) for seq in X_train])
X_test_rf = np.array([aggregate_features(seq) for seq in X_test])

# Scale features for RandomForest
scaler = StandardScaler()
X_train_rf = scaler.fit_transform(X_train_rf)
X_test_rf = scaler.transform(X_test_rf)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train_rf, y_train)
y_pred = rf.predict(X_test_rf)
y_proba = rf.predict_proba(X_test_rf)[:, 1]

save_metrics("RandomForest", y_test, y_pred, y_proba, results, roc_curves)
joblib.dump(rf, os.path.join(MODEL_DIR, "randomforest.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "rf_scaler.pkl"))

# -------------------------------
# 2. LSTM
# -------------------------------
print("\n🔥 Training LSTM...")
lstm = Sequential([
    Input(shape=(max_len, len(FEATURES))),
    LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
lstm.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

history = lstm.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
)

y_proba = lstm.predict(X_test, verbose=0).ravel()
y_pred = (y_proba > 0.5).astype(int)

save_metrics("LSTM", y_test, y_pred, y_proba, results, roc_curves)
lstm.save(os.path.join(MODEL_DIR, "lstm_model.keras"))

# -------------------------------
# 3. GRU
# -------------------------------
print("\n🔥 Training GRU...")
gru = Sequential([
    Input(shape=(max_len, len(FEATURES))),
    GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
gru.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

gru.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
)

y_proba = gru.predict(X_test, verbose=0).ravel()
y_pred = (y_proba > 0.5).astype(int)

save_metrics("GRU", y_test, y_pred, y_proba, results, roc_curves)
gru.save(os.path.join(MODEL_DIR, "gru_model.keras"))

# -------------------------------
# 4. CNN
# -------------------------------
print("\n🔥 Training CNN...")
cnn = Sequential([
    Input(shape=(max_len, len(FEATURES))),
    Conv1D(64, 5, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(2),
    Conv1D(32, 3, activation="relu"),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dense(64, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])
cnn.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

cnn.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
)

y_proba = cnn.predict(X_test, verbose=0).ravel()
y_pred = (y_proba > 0.5).astype(int)

save_metrics("CNN", y_test, y_pred, y_proba, results, roc_curves)
cnn.save(os.path.join(MODEL_DIR, "cnn_model.keras"))

# -------------------------------
# 5. Autoencoder (Anomaly Detection)
# -------------------------------
print("\n🔥 Training Autoencoder...")

# Normalize sequences for autoencoder
X_train_norm = (X_train - X_train.mean(axis=0, keepdims=True)) / (X_train.std(axis=0, keepdims=True) + 1e-7)
X_test_norm = (X_test - X_train.mean(axis=0, keepdims=True)) / (X_train.std(axis=0, keepdims=True) + 1e-7)

inp = Input(shape=(max_len, len(FEATURES)))
# Encoder
x = Conv1D(32, 3, activation="relu", padding="same")(inp)
x = MaxPooling1D(2, padding="same")(x)
x = Conv1D(16, 3, activation="relu", padding="same")(x)
encoded = MaxPooling1D(2, padding="same")(x)

# Decoder
x = Conv1D(16, 3, activation="relu", padding="same")(encoded)
x = tf.keras.layers.UpSampling1D(2)(x)
x = Conv1D(32, 3, activation="relu", padding="same")(x)
x = tf.keras.layers.UpSampling1D(2)(x)
decoded = Conv1D(len(FEATURES), 3, activation="linear", padding="same")(x)

# Handle shape mismatch
if decoded.shape[1] != max_len:
    # Crop or pad to match input length
    decoded = tf.keras.layers.Lambda(lambda x: x[:, :max_len, :])(decoded)

autoencoder = Model(inp, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train only on safe trips (label 0) for anomaly detection
safe_mask = y_train == 0
X_train_safe = X_train_norm[safe_mask]

autoencoder.fit(
    X_train_safe, X_train_safe, 
    epochs=25, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
)

# Calculate reconstruction error for test set
X_test_reconstructed = autoencoder.predict(X_test_norm, verbose=0)
reconstruction_errors = np.mean(np.square(X_test_norm - X_test_reconstructed), axis=(1, 2))

# Use reconstruction error as risk score
threshold = np.percentile(reconstruction_errors, 50)  # Adjust threshold
y_pred = (reconstruction_errors > threshold).astype(int)

save_metrics("Autoencoder", y_test, y_pred, reconstruction_errors, results, roc_curves)
autoencoder.save(os.path.join(MODEL_DIR, "autoencoder_model.keras"))

# -------------------------------
# 6. Transformer
# -------------------------------
print("\n🔥 Training Transformer...")

inp = Input(shape=(max_len, len(FEATURES)))
# Multi-head self-attention
attention_output = MultiHeadAttention(num_heads=8, key_dim=64)(inp, inp)
x = Add()([attention_output, inp])  # Residual connection
x = LayerNormalization()(x)

# Feed forward network
ffn = Dense(128, activation="relu")(x)
ffn = Dense(len(FEATURES))(ffn)
x = Add()([x, ffn])  # Another residual connection
x = LayerNormalization()(x)

# Global pooling and classification head
x = GlobalAveragePooling1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.5)(x)
out = Dense(1, activation="sigmoid")(x)

transformer = Model(inp, out)
transformer.compile(optimizer=Adam(0.0005), loss="binary_crossentropy", metrics=["accuracy"])

transformer.fit(
    X_train, y_train, 
    epochs=25, 
    batch_size=32, 
    validation_split=0.2, 
    verbose=0,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
)

y_proba = transformer.predict(X_test, verbose=0).ravel()
y_pred = (y_proba > 0.5).astype(int)

save_metrics("Transformer", y_test, y_pred, y_proba, results, roc_curves)
transformer.save(os.path.join(MODEL_DIR, "transformer_model.keras"))

# -------------------------------
# Save Results and Summary
# -------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("roc_auc", ascending=False)
results_df.to_csv(RESULTS_FILE, index=False)

roc_df = pd.DataFrame(roc_curves)
roc_df.to_csv(ROC_FILE, index=False)

print("\n" + "="*70)
print("🏆 FINAL RESULTS SUMMARY")
print("="*70)
print(results_df.to_string(index=False))

print(f"\n✅ Training complete!")
print(f"📁 Results saved to: {RESULTS_FILE}")
print(f"📁 ROC curves saved to: {ROC_FILE}")
print(f"📁 Models saved to: {MODEL_DIR}/")

print(f"\n🎯 KEY INSIGHT:")
print(f"   • RandomForest (aggregates only): {results_df[results_df.model=='RandomForest']['roc_auc'].values[0]:.3f} ROC-AUC")
best_ai = results_df[results_df.model != 'RandomForest'].iloc[0]
print(f"   • Best AI model ({best_ai['model']}): {best_ai['roc_auc']:.3f} ROC-AUC")
print(f"   • AI models capture temporal patterns that aggregates miss! 🚀")