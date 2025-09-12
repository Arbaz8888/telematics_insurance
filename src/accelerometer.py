# scripts/generate_accelerometer.py

import pandas as pd
import numpy as np
import os

# Load trips and telematics to align accelerometer with timestamps
telematics = pd.read_csv("data/telematics_data.csv")
telematics["timestamp"] = pd.to_datetime(telematics["timestamp"])

os.makedirs("data", exist_ok=True)

def simulate_accelerometer(trip_id, trip_data):
    n = len(trip_data)
    rng = np.random.default_rng(seed=42)

    # Simulated raw accelerometer signals
    accel_x = rng.normal(0, 0.05, n)  # lateral (side-to-side)
    accel_y = rng.normal(0, 0.05, n)  # forward-backward
    accel_z = rng.normal(1, 0.02, n)  # gravity baseline ~1g

    # Braking g-force: based on speed changes
    speed = trip_data["speed_kmh"].to_numpy()
    braking_g = np.abs(np.diff(np.insert(speed, 0, speed[0]))) / 100.0
    braking_g = np.clip(braking_g + rng.normal(0, 0.05, n), 0, 2)

    # Cornering g-force: sinusoidal variation + noise
    cornering_g = np.abs(np.sin(np.linspace(0, 3 * np.pi, n)) * 0.5 + rng.normal(0, 0.05, n))
    cornering_g = np.clip(cornering_g, 0, 2)

    return pd.DataFrame({
        "trip_id": trip_id,
        "timestamp": trip_data["timestamp"].values,
        "accel_x": accel_x,
        "accel_y": accel_y,
        "accel_z": accel_z,
        "braking_g": braking_g,
        "cornering_g": cornering_g
    })

# Generate for all trips
accel_all = []
for trip_id, group in telematics.groupby("trip_id"):
    accel_all.append(simulate_accelerometer(trip_id, group))

accelerometer_df = pd.concat(accel_all, ignore_index=True)

# Save to CSV
output_path = "data/accelerometer.csv"
accelerometer_df.to_csv(output_path, index=False)

print(f"✅ Accelerometer data saved at {output_path}")
print(accelerometer_df.head())
