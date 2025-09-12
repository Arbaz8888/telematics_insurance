import random

def calculate_contextual_risk(row):
    """
    Given a row from driver_features.csv, return contextual risk adjustments.
    """
    adjustments = []
    penalty = 0

    # Weather simulation
    weather = random.choice(["Clear", "Rain", "Fog", "Snow"])
    if weather == "Rain":
        penalty += 10
        adjustments.append("🌧 Rain (+10%)")
    elif weather == "Fog":
        penalty += 7
        adjustments.append("🌫 Fog (+7%)")
    elif weather == "Snow":
        penalty += 15
        adjustments.append("❄️ Snow (+15%)")
    else:
        adjustments.append("☀️ Clear (0%)")

    # Accident hotspot adjustment
    if row.get("accident_rate_area", 0) > 0.07:
        penalty += 8
        adjustments.append("🚨 Accident-prone area (+8%)")

    # Crime hotspot adjustment
    if row.get("crime_index_area", 0) > 0.1:
        penalty += 5
        adjustments.append("⚖️ Crime-prone area (+5%)")

    return penalty, adjustments, weather
