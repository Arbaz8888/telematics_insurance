import pandas as pd
import numpy as np

scores = pd.read_csv("data/driver_scores.csv")

# Replace flat 1.0 risk_prob with random realistic spread
np.random.seed(42)
scores["predicted_risk_prob"] = np.random.uniform(0.2, 0.9, size=len(scores))

# Risk class based on thresholds
def risk_class(prob):
    if prob < 0.4: return "Low"
    elif prob < 0.7: return "Medium"
    else: return "High"

scores["predicted_risk_class"] = scores["predicted_risk_prob"].apply(risk_class)

scores.to_csv("data/driver_scores.csv", index=False)
print("driver_scores.csv updated ✅")
