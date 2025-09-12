# generate_gamification.py
import pandas as pd
import random
import os

# Load drivers.csv to get the correct driver IDs
drivers = pd.read_csv("data/drivers.csv")

# Prepare gamification records
records = []
for _, row in drivers.iterrows():
    driver_id = row["driver_id"]

    safe_streak = random.randint(0, 20)       # days of safe driving streak
    reward_points = random.randint(100, 5000) # reward points balance
    badges = []

    # Assign badges based on thresholds
    if safe_streak >= 10:
        badges.append("Consistent Safe Driver")
    if reward_points > 3000:
        badges.append("Loyalty Star")
    if safe_streak >= 5 and random.random() < 0.3:
        badges.append("No Phone Week")

    records.append({
        "driver_id": driver_id,
        "safe_driving_streak": safe_streak,
        "reward_points": reward_points,
        "badges_earned": str(badges)  # store as stringified list
    })

# Save regenerated gamification CSV
os.makedirs("data", exist_ok=True)
output_path = "data/driver_gamification.csv"
pd.DataFrame(records).to_csv(output_path, index=False)

print(f"✅ Regenerated gamification file with {len(records)} records at {output_path}")
