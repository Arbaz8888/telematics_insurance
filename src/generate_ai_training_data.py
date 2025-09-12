"""
File: generate_ai_training_data.py
Purpose: Generate synthetic telematics trip-level data for AI model training.
         Produces a balanced dataset of risky vs safe trips.
Author: Arbaz Attar
"""

import pandas as pd
import numpy as np
import os

# -------------------------------
# Configurations
# -------------------------------
OUTPUT_FILE = "data/ai_training_data.csv"
N_TRIPS = 1000   # total number of trips to generate
RISK_RATIO = 0.5 # proportion of risky trips

np.random.seed(42)  # reproducibility

os.makedirs("data", exist_ok=True)

# -------------------------------
# Data Generation
# -------------------------------
def generate_trip_data(n_trips, risk_ratio=0.5):
    n_risky = int(n_trips * risk_ratio)
    n_safe = n_trips - n_risky

    trips = []

    # Safe trips
    for _ in range(n_safe):
        avg_speed = np.random.normal(40, 5)      # average speed ~40 km/h
        max_speed = np.random.normal(70, 10)     # max speed lower
        braking_events = np.random.poisson(2)    # light braking
        cornering_events = np.random.poisson(1)  # few sharp turns
        trip_duration = np.random.normal(30, 5)  # duration in minutes
        trip_risk_label = 0
        trips.append([avg_speed, max_speed, braking_events,
                      cornering_events, trip_duration, trip_risk_label])

    # Risky trips
    for _ in range(n_risky):
        avg_speed = np.random.normal(60, 8)      # higher avg speed
        max_speed = np.random.normal(120, 15)    # very high speed
        braking_events = np.random.poisson(6)    # frequent braking
        cornering_events = np.random.poisson(4)  # more sharp turns
        trip_duration = np.random.normal(25, 7)  # shorter but erratic
        trip_risk_label = 1
        trips.append([avg_speed, max_speed, braking_events,
                      cornering_events, trip_duration, trip_risk_label])

    # Shuffle to mix risky and safe
    np.random.shuffle(trips)

    return pd.DataFrame(trips, columns=[
        "avg_speed", "max_speed", "braking_events",
        "cornering_events", "trip_duration", "trip_risk_label"
    ])

# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    df = generate_trip_data(N_TRIPS, RISK_RATIO)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Generated {len(df)} trips and saved to {OUTPUT_FILE}")
    print(df['trip_risk_label'].value_counts())
