import os
import json
import time
import pandas as pd

STREAM_PATH = "data/stream_buffer"
OUTPUT_PATH = "data/stream_processed.csv"

# -------------------------------
# Load trips for trip_id → driver_id mapping
# -------------------------------
trips = pd.read_csv("data/trips.csv")
trip_map = dict(zip(trips["trip_id"], trips["driver_id"]))

# -------------------------------
# Helper: Update rolling driver stats
# -------------------------------
def update_scores(batch, scores_df):
    for event in batch:
        trip_id = event.get("trip_id")
        driver_id = trip_map.get(trip_id)

        if driver_id is None:
            continue  # skip events we can't map

        speed = event.get("speed_kmh", 0)
        accel = event.get("acceleration_ms2", 0)

        # Initialize driver if not exists
        if driver_id not in scores_df.index:
            scores_df.loc[driver_id] = {
                "events_processed": 0,
                "hard_brakes": 0,
                "speeding": 0,
                "avg_speed": 0.0,
                "rolling_risk_score": 0.0
            }

        # Update counts
        scores_df.loc[driver_id, "events_processed"] += 1
        if accel < -3:  # harsh deceleration threshold
            scores_df.loc[driver_id, "hard_brakes"] += 1
        if speed > 100:  # speeding threshold
            scores_df.loc[driver_id, "speeding"] += 1

        # Update average speed (incremental mean)
        prev_avg = scores_df.loc[driver_id, "avg_speed"]
        n = scores_df.loc[driver_id, "events_processed"]
        scores_df.loc[driver_id, "avg_speed"] = (prev_avg * (n - 1) + speed) / n

        # Risk score heuristic
        risk = (
            scores_df.loc[driver_id, "hard_brakes"] * 5 +
            scores_df.loc[driver_id, "speeding"] * 3 +
            (scores_df.loc[driver_id, "avg_speed"] / 10)
        )
        scores_df.loc[driver_id, "rolling_risk_score"] = round(risk, 2)

    return scores_df


# -------------------------------
# Main Streaming Processor
# -------------------------------
print("⚡ Starting Stream Processor...")

processed_files = set()
scores_df = pd.DataFrame(columns=[
    "events_processed", "hard_brakes", "speeding", "avg_speed", "rolling_risk_score"
])
scores_df.index.name = "driver_id"

while True:
    files = sorted(os.listdir(STREAM_PATH))
    new_files = [f for f in files if f.endswith(".json") and f not in processed_files]

    if not new_files:
        time.sleep(1)
        continue

    for file in new_files:
        file_path = os.path.join(STREAM_PATH, file)
        with open(file_path, "r") as f:
            batch = json.load(f)

        scores_df = update_scores(batch, scores_df)
        processed_files.add(file)

        # Save rolling scores to CSV
        scores_df.to_csv(OUTPUT_PATH)

        print(f"✅ Processed {file}, updated {len(scores_df)} drivers → {OUTPUT_PATH}")

    time.sleep(1)
