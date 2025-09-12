import pandas as pd

# Load files
drivers = pd.read_csv("data/drivers.csv")
scores = pd.read_csv("data/stream_processed.csv")

# Merge names into scores
scores_named = scores.merge(
    drivers[["driver_id", "name"]], on="driver_id", how="left"
)

# Save back (overwrite the original)
scores_named.to_csv("data/stream_processed.csv", index=False)

print("✅ stream_processed.csv regenerated with driver names included.")
print(scores_named.head())
