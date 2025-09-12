import pandas as pd
import os

# Load policies and driver scores
policies = pd.read_csv("data/policies.csv")
scores = pd.read_csv("data/driver_scores.csv")

# Merge policies with risk scores
df = policies.merge(scores[["driver_id", "predicted_risk_prob"]], on="driver_id", how="left")

# Pricing adjustment formula
# Safe drivers (<0.3) get discounts, risky drivers pay more
def adjust_premium(base, risk_prob, alpha=0.5, beta=0.2):
    if pd.isna(risk_prob):
        return base  # no score available, keep original
    if risk_prob < 0.3:
        return round(base * (1 - beta * (0.3 - risk_prob)), 2)  # discount
    else:
        return round(base * (1 + alpha * risk_prob), 2)          # surcharge

df["adjusted_premium"] = df.apply(
    lambda row: adjust_premium(row["premium_amount"], row["predicted_risk_prob"]),
    axis=1
)

# Save updated premiums
os.makedirs("data", exist_ok=True)
output_path = "data/adjusted_policies.csv"
df.to_csv(output_path, index=False)

print("Premiums adjusted and saved at:", output_path)
print(df[["driver_id", "premium_amount", "predicted_risk_prob", "adjusted_premium"]].head())
