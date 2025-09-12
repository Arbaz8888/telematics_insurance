# File: src/ai_api.py
# Purpose: Secure API for AI risk prediction (Transformer)
# Run: uvicorn src.ai_api:app --reload --port 8000

import os
import tempfile
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from cryptography.fernet import Fernet
import json

# -------------------------------
# Config
# -------------------------------
MODEL_PATH = "models/transformer_model.keras.enc"   # encrypted file
KEY_PATH = "models/encryption.key"

# Load encryption key
if not os.path.exists(KEY_PATH):
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)

with open(KEY_PATH, "rb") as f:
    SECRET_KEY = f.read()

fernet = Fernet(SECRET_KEY)

# -------------------------------
# Secure Model Loading
# -------------------------------
print("🔐 Decrypting Transformer model at runtime...")

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        enc_bytes = f.read()
    dec_bytes = fernet.decrypt(enc_bytes)

    # Write decrypted model to a temp file
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp.write(dec_bytes)
        tmp_path = tmp.name

    model = tf.keras.models.load_model(tmp_path)
    print("✅ Transformer model loaded securely.")
else:
    print("⚠️ Encrypted model not found. Falling back to unencrypted path.")
    model = tf.keras.models.load_model("models/transformer_model.keras")

# -------------------------------
# FastAPI app
# -------------------------------
app = FastAPI(title="AI Risk Prediction API", version="1.0")

class TripData(BaseModel):
    trip_features: list  # [[timestep1_features], [timestep2_features], ...]

MAX_LEN = 299  # must match training

@app.post("/predict_risk")
def predict_risk(data: TripData):
    try:
        # Encrypt/decrypt simulation
        raw = str(data.trip_features).encode()
        encrypted = fernet.encrypt(raw)
        decrypted = fernet.decrypt(encrypted).decode()

        # Parse JSON safely
        X = np.array(json.loads(decrypted.replace("'", '"')))  # (timesteps, 5)

        # Pad to MAX_LEN
        if X.shape[0] < MAX_LEN:
            pad_len = MAX_LEN - X.shape[0]
            X = np.pad(X, ((0, pad_len), (0, 0)), mode="constant")
        elif X.shape[0] > MAX_LEN:
            X = X[:MAX_LEN, :]  # truncate if longer

        X = np.expand_dims(X, axis=0)  # (1, MAX_LEN, 5)

        # Predict
        risk_score = float(model.predict(X, verbose=0).ravel()[0])
        risk_label = "High" if risk_score > 0.5 else "Low"

        return {"risk_score": round(risk_score, 4), "risk_label": risk_label}

    except Exception as e:
        return {"error": str(e)}

