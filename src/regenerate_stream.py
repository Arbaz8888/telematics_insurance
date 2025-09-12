import pandas as pd

# Load files
drivers = pd.read_csv("data/drivers.csv")
features = pd.read_csv("data/driver_features_decrypted.csv")

# Map driver_id in order
features["driver_id"] = drivers["driver_id"]

# Drop anon_id
features = features.drop(columns=["anon_id"], errors="ignore")

# Reorder so driver_id is first
cols = ["driver_id"] + [c for c in features.columns if c != "driver_id"]
features = features[cols]

# Save back
features.to_csv("data/driver_features_decrypted.csv", index=False)

print("✅ Fixed driver_features_decrypted.csv with driver_id")
