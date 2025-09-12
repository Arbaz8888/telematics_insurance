import pandas as pd
import numpy as np
import uuid

# Number of synthetic drivers to generate
N = 300

# Random seed for reproducibility
np.random.seed(42)

def random_gender():
    return np.random.choice(["M", "F", "Other"], p=[0.45, 0.45, 0.1])

rows = []
for _ in range(N):
    driver_id = str(uuid.uuid4())
    age = np.random.randint(18, 70)
    gender = random_gender()
    total_trips = np.random.randint(1, 800)

    avg_distance = np.random.uniform(20, 120)  # km
    avg_duration = np.random.uniform(30, 150)  # minutes
    night_driving_pct = np.random.uniform(0, 1)
    route_familiarity = np.random.uniform(0.4, 0.9)

    count_DISTRACTION = np.random.poisson(50 * night_driving_pct)
    count_HARD_BRAKE = np.random.poisson(20 + 0.05 * total_trips)
    count_SPEEDING = np.random.poisson(15 + 0.05 * total_trips)

    crime_index_area = np.random.uniform(20, 90)
    accident_rate_area = np.random.uniform(0.05, 0.2)

    avg_speed = np.random.uniform(50, 90) + (count_SPEEDING * 0.02)
    max_speed = avg_speed + np.random.uniform(10, 40)

    avg_accel = np.random.uniform(-0.1, 0.1)
    avg_braking = np.random.uniform(0.18, 0.22)
    avg_cornering = np.random.uniform(0.18, 0.22)

    # Risk score = influenced by braking, speeding, night driving
    risk_score = min(1, max(0,
        0.3*night_driving_pct +
        0.3*(count_SPEEDING/ (total_trips+1)) +
        0.2*(count_HARD_BRAKE/ (total_trips+1)) +
        0.2*(crime_index_area/100)
    ))

    rows.append([
        driver_id, age, gender, round(risk_score, 2), total_trips,
        round(avg_distance, 2), round(avg_duration, 2),
        round(night_driving_pct, 2), round(route_familiarity, 2),
        count_DISTRACTION, count_HARD_BRAKE, count_SPEEDING,
        round(crime_index_area, 2), round(accident_rate_area, 3),
        round(avg_speed, 2), round(max_speed, 2),
        round(avg_accel, 3), round(avg_braking, 3), round(avg_cornering, 3)
    ])

# Build dataframe
cols = [
    "driver_id","age","gender","risk_score","total_trips",
    "avg_distance","avg_duration","night_driving_pct","route_familiarity",
    "count_DISTRACTION","count_HARD_BRAKE","count_SPEEDING",
    "crime_index_area","accident_rate_area","avg_speed","max_speed",
    "avg_accel","avg_braking","avg_cornering"
]
df = pd.DataFrame(rows, columns=cols)

# Save to /data
output_path = "data/driver_features.csv"
df.to_csv(output_path, index=False)
print(f"✅ Synthetic driver_features.csv generated with {N} rows at {output_path}")
