import pandas as pd

# Load datasets
drivers = pd.read_csv("data/drivers.csv")
trips = pd.read_csv("data/trips.csv")
events = pd.read_csv("data/events.csv")
external = pd.read_csv("data/external_factors.csv")
telematics = pd.read_csv("data/telematics_data.csv")

# -------------------------------
# Trip-level aggregation
# -------------------------------
trip_features = trips.groupby("driver_id").agg(
    total_trips=("trip_id", "count"),
    avg_distance=("distance_km", "mean"),
    avg_duration=("duration_min", "mean"),
    night_driving_pct=("night_driving_percentage", "mean"),
    route_familiarity=("route_familiarity_score", "mean"),
).reset_index()

# -------------------------------
# Event-level aggregation (link via trips)
# -------------------------------
events_with_driver = events.merge(trips[["trip_id", "driver_id"]], on="trip_id", how="left")
event_counts = events_with_driver.groupby(["driver_id", "event_type"]).size().unstack(fill_value=0)
event_counts = event_counts.add_prefix("count_").reset_index()


# -------------------------------
# External factors aggregation
# -------------------------------
external_features = external.groupby("trip_id").mean(numeric_only=True).reset_index()
external_features = trips[["trip_id", "driver_id"]].merge(external_features, on="trip_id")
external_features = external_features.groupby("driver_id").mean(numeric_only=True).reset_index()

# -------------------------------
# Telematics aggregation
# -------------------------------
telematics_features = telematics.groupby("trip_id").agg(
    avg_speed=("speed_kmh", "mean"),
    max_speed=("speed_kmh", "max"),
    avg_accel=("acceleration_ms2", "mean"),
    avg_braking=("braking_intensity", "mean"),
    avg_cornering=("cornering_gforce", "mean"),
).reset_index()

telematics_features = trips[["trip_id", "driver_id"]].merge(telematics_features, on="trip_id")
telematics_features = telematics_features.groupby("driver_id").mean(numeric_only=True).reset_index()

# -------------------------------
# Merge everything into driver-level dataset
# -------------------------------
driver_features = drivers[["driver_id", "age", "gender", "risk_score"]] \
    .merge(trip_features, on="driver_id", how="left") \
    .merge(event_counts, on="driver_id", how="left") \
    .merge(external_features, on="driver_id", how="left") \
    .merge(telematics_features, on="driver_id", how="left")

# Fill missing event types with 0
driver_features = driver_features.fillna(0)

print("Driver-level feature set created!")
print(driver_features.head())

# Save for model training
driver_features.to_csv("data/driver_features.csv", index=False)
