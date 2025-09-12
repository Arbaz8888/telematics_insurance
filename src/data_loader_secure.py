import pandas as pd
import os
from cryptography.fernet import Fernet

# -------------------------------
# Setup: Encryption Key
# -------------------------------
os.makedirs("keys", exist_ok=True)
key_path = "keys/fernet.key"

if not os.path.exists(key_path):
    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
else:
    with open(key_path, "rb") as f:
        key = f.read()

fernet = Fernet(key)

# -------------------------------
# Load Drivers (with consent enforcement + anonymization)
# -------------------------------
# Load full engineered features instead
features = pd.read_csv("data/driver_features.csv")

# Load drivers to check consent
drivers = pd.read_csv("data/drivers.csv")
consented_ids = drivers.loc[drivers["consent_status"] == True, "driver_id"]

# Enforce consent
features = features[features["driver_id"].isin(consented_ids)].copy()

# Anonymize driver IDs (replace with pseudonyms)
anon_map = {d: f"Driver-{i:04d}" for i, d in enumerate(features["driver_id"].unique())}
features["anon_id"] = features["driver_id"].map(anon_map)
features = features.drop(columns=["driver_id"])  # remove raw IDs

# Save anonymized features
output_path = "data/driver_features_secure.csv"
features.to_csv(output_path, index=False)


# -------------------------------
# Encrypt file
# -------------------------------
with open(output_path, "rb") as f:
    encrypted_data = fernet.encrypt(f.read())

enc_path = output_path + ".enc"
with open(enc_path, "wb") as f:
    f.write(encrypted_data)

print(f"🔐 Encrypted driver features saved at {enc_path}")

# -------------------------------
# Example: Decryption for later use
# -------------------------------
with open(enc_path, "rb") as f:
    encrypted = f.read()

decrypted_data = fernet.decrypt(encrypted)

dec_path = "data/driver_features_decrypted.csv"
with open(dec_path, "wb") as f:
    f.write(decrypted_data)

print(f"🔓 Decrypted copy written back at {dec_path}")
