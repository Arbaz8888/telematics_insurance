# generate_drivers_csv.py

import pandas as pd
import random

# Load driver_features.csv
features = pd.read_csv("data/driver_features.csv")

# Load gamification to preserve matching driver_id
try:
    gamification = pd.read_csv("data/driver_gamification.csv")
    gamification_ids = set(gamification["driver_id"])
except:
    gamification = None
    gamification_ids = set()

# Generate names
first_names = ["Alice", "Bob", "Charlie", "David", "Eva", "Frank", "Grace", "Helen", "Ivy", "Jack"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Wilson", "Moore"]

def random_name():
    return random.choice(first_names) + " " + random.choice(last_names)

# Build drivers.csv
drivers = pd.DataFrame({
    "driver_id": features["driver_id"],   # exact IDs from features
    "name": [random_name() for _ in range(len(features))],
    "age": features["age"],
    "gender": features["gender"]
})

# Check consistency with gamification
if gamification is not None:
    missing_ids = gamification_ids - set(drivers["driver_id"])
    if missing_ids:
        print("⚠️ Warning: Some gamification driver_ids missing in drivers.csv:", missing_ids)

# Save
drivers.to_csv("data/drivers.csv", index=False)

print("✅ drivers.csv regenerated with", len(drivers), "rows")
print(drivers.head())
