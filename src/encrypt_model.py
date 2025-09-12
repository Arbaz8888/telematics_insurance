# File: src/encrypt_model.py
# Purpose: Encrypt Transformer model (.keras) into .enc format

import os
from cryptography.fernet import Fernet

MODEL_PATH = "models/transformer_model.keras"
ENC_MODEL_PATH = "models/transformer_model.keras.enc"
KEY_PATH = "models/encryption.key"

# Load or generate key
if not os.path.exists(KEY_PATH):
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)

with open(KEY_PATH, "rb") as f:
    SECRET_KEY = f.read()

fernet = Fernet(SECRET_KEY)

# Read raw model bytes
with open(MODEL_PATH, "rb") as f:
    model_bytes = f.read()

# Encrypt
enc_bytes = fernet.encrypt(model_bytes)

# Save encrypted model
with open(ENC_MODEL_PATH, "wb") as f:
    f.write(enc_bytes)

print(f"✅ Encrypted model saved to {ENC_MODEL_PATH}")
