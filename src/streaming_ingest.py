import time
import pandas as pd
import json
import os

# -------------------------------
# Config
# -------------------------------
DATA_PATH = "data/telematics_data.csv"
STREAM_PATH = "data/stream_buffer"
BATCH_SIZE = 10   # number of rows per batch (like Kafka messages)
SLEEP_TIME = 1    # seconds between batches

os.makedirs(STREAM_PATH, exist_ok=True)

# -------------------------------
# Load Telematics Data
# -------------------------------
df = pd.read_csv(DATA_PATH)

# Sort by trip + timestamp for realistic ordering
df = df.sort_values(["trip_id", "timestamp"]).reset_index(drop=True)

# -------------------------------
# Streaming Generator
# -------------------------------
def stream_events(df, batch_size=BATCH_SIZE):
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i + batch_size].to_dict(orient="records")

# -------------------------------
# Main Ingestion Loop
# -------------------------------
print("🚦 Starting Telematics Streaming Simulation...")

for batch_id, batch in enumerate(stream_events(df)):
    # Save each batch as a JSON file (simulating Kafka → S3)
    file_path = os.path.join(STREAM_PATH, f"batch_{batch_id}.json")
    with open(file_path, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"📤 Streamed batch {batch_id} with {len(batch)} events → {file_path}")
    time.sleep(SLEEP_TIME)

print("✅ Streaming simulation complete.")
