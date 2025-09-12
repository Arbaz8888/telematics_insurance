import pandas as pd

# Load CSVs
drivers = pd.read_csv("data/drivers.csv")
vehicles = pd.read_csv("data/vehicles.csv")
policies = pd.read_csv("data/policies.csv")
trips = pd.read_csv("data/trips.csv")
telematics = pd.read_csv("data/telematics_data.csv")
events = pd.read_csv("data/events.csv")
external = pd.read_csv("data/external_factors.csv")
biometrics = pd.read_csv("data/biometrics.csv")
gamification = pd.read_csv("data/driver_gamification.csv")


# Quick sanity checks
print("Drivers:", drivers.shape)
print("Vehicles:", vehicles.shape)
print("Policies:", policies.shape)
print("Trips:", trips.shape)
print("Telematics:", telematics.shape)
print("Events:", events.shape)
print("External Factors:", external.shape)
print("Biometrics:", biometrics.shape)
print("Gamification:", gamification.shape)

# Example preview
print("\nSample Drivers:\n", drivers.head())
print("\nSample Trips:\n", trips.head())
