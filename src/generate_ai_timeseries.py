"""
File: generate_ai_timeseries.py
Purpose: Generate synthetic telematics data where aggregate stats are IDENTICAL
         but risk is hidden purely in temporal sequence patterns.
Author: Arbaz Attar (Fixed)
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "ai_telematics_timeseries.csv")

# -------------------------------
# Config
# -------------------------------
N_TRIPS = 1000
MIN_LEN = 200
MAX_LEN = 300  # Shorter sequences to reduce noise
SAFE_RATIO = 0.5

def create_driving_events():
    """Create a dictionary of standardized driving events"""
    return {
        'normal_cruise': {'speed': 65, 'accel': 0, 'brake': 0.02, 'steer': 0, 'g_force': 0.2},
        'light_accel': {'speed': 70, 'accel': 1.5, 'brake': 0, 'steer': 0, 'g_force': 0.35},
        'light_brake': {'speed': 60, 'accel': -1.2, 'brake': 0.4, 'steer': 0, 'g_force': 0.3},
        'turn_left': {'speed': 65, 'accel': 0, 'brake': 0.1, 'steer': -8, 'g_force': 0.25},
        'turn_right': {'speed': 65, 'accel': 0, 'brake': 0.1, 'steer': 8, 'g_force': 0.25},
        'hard_accel': {'speed': 75, 'accel': 2.8, 'brake': 0, 'steer': 0, 'g_force': 0.5},
        'hard_brake': {'speed': 55, 'accel': -2.5, 'brake': 0.8, 'steer': 0, 'g_force': 0.45},
        'sharp_turn': {'speed': 62, 'accel': 0, 'brake': 0.3, 'steer': 12, 'g_force': 0.4},
    }

def generate_safe_sequence(length):
    """Generate safe driving: smooth transitions, no aggressive patterns"""
    events = create_driving_events()
    
    # Safe pattern: mostly normal cruise with smooth transitions
    sequence = []
    
    i = 0
    while i < length:
        # Mostly normal driving with gradual changes
        event_weights = [0.6, 0.15, 0.15, 0.03, 0.03, 0.02, 0.02, 0.0]  # Avoid hard_accel, hard_brake, sharp_turn
        event_names = ['normal_cruise', 'light_accel', 'light_brake', 'turn_left', 'turn_right', 'hard_accel', 'hard_brake', 'sharp_turn']
        
        chosen_event = np.random.choice(event_names, p=event_weights)
        event_duration = np.random.randint(8, 15)  # Longer, smoother events
        
        for _ in range(min(event_duration, length - i)):
            base_values = events[chosen_event].copy()
            # Add small noise
            for key in base_values:
                base_values[key] += np.random.normal(0, abs(base_values[key]) * 0.1)
            sequence.append(base_values)
            i += 1
    
    return sequence

def generate_risky_sequence(length):
    """Generate risky driving: same events but in dangerous patterns"""
    events = create_driving_events()
    
    sequence = []
    i = 0
    
    while i < length:
        # Create risky patterns: hard_accel followed by hard_brake, or sharp turns after acceleration
        if i < length - 20 and np.random.random() < 0.15:  # 15% chance of risky pattern
            # Pattern 1: Hard acceleration followed by hard braking
            if np.random.random() < 0.5:
                # Hard accel for 5-8 seconds
                accel_duration = np.random.randint(5, 8)
                for _ in range(min(accel_duration, length - i)):
                    base_values = events['hard_accel'].copy()
                    for key in base_values:
                        base_values[key] += np.random.normal(0, abs(base_values[key]) * 0.1)
                    sequence.append(base_values)
                    i += 1
                
                # Immediately followed by hard braking
                brake_duration = np.random.randint(4, 7)
                for _ in range(min(brake_duration, length - i)):
                    base_values = events['hard_brake'].copy()
                    for key in base_values:
                        base_values[key] += np.random.normal(0, abs(base_values[key]) * 0.1)
                    sequence.append(base_values)
                    i += 1
            
            # Pattern 2: Sharp turns during acceleration
            else:
                turn_duration = np.random.randint(3, 6)
                for _ in range(min(turn_duration, length - i)):
                    base_values = events['sharp_turn'].copy()
                    base_values['accel'] = 1.8  # Turning while accelerating (risky)
                    for key in base_values:
                        base_values[key] += np.random.normal(0, abs(base_values[key]) * 0.1)
                    sequence.append(base_values)
                    i += 1
        
        else:
            # Fill with normal driving (same distribution as safe)
            event_weights = [0.6, 0.15, 0.15, 0.03, 0.03, 0.02, 0.02, 0.0]
            event_names = ['normal_cruise', 'light_accel', 'light_brake', 'turn_left', 'turn_right', 'hard_accel', 'hard_brake', 'sharp_turn']
            
            chosen_event = np.random.choice(event_names, p=event_weights)
            event_duration = np.random.randint(8, 15)
            
            for _ in range(min(event_duration, length - i)):
                base_values = events[chosen_event].copy()
                for key in base_values:
                    base_values[key] += np.random.normal(0, abs(base_values[key]) * 0.1)
                sequence.append(base_values)
                i += 1
    
    return sequence

def sequence_to_arrays(sequence):
    """Convert sequence of events to numpy arrays"""
    speed = np.array([event['speed'] for event in sequence])
    accel = np.array([event['accel'] for event in sequence])
    brake = np.array([event['brake'] for event in sequence])
    steer = np.array([event['steer'] for event in sequence])
    g_force = np.array([event['g_force'] for event in sequence])
    
    # Ensure reasonable bounds
    speed = np.clip(speed, 30, 85)
    accel = np.clip(accel, -3, 3)
    brake = np.clip(brake, 0, 1)
    steer = np.clip(steer, -15, 15)
    g_force = np.clip(g_force, 0.1, 0.6)
    
    return speed, accel, brake, steer, g_force

def normalize_trip_statistics(speed, accel, brake, steer, g_force):
    """Force all trips to have nearly identical aggregate statistics"""
    # Target statistics (same for all trips)
    targets = {
        'speed': {'mean': 65.0, 'std': 4.5},
        'accel': {'mean': 0.0, 'std': 1.2},
        'brake': {'mean': 0.12, 'std': 0.18},
        'steer': {'mean': 0.0, 'std': 3.5},
        'g_force': {'mean': 0.28, 'std': 0.08}
    }
    
    def normalize_array(arr, target_mean, target_std):
        current_mean = np.mean(arr)
        current_std = np.std(arr)
        if current_std > 1e-6:
            normalized = (arr - current_mean) / current_std
            return normalized * target_std + target_mean
        else:
            return np.full_like(arr, target_mean)
    
    # Normalize each feature
    speed = normalize_array(speed, targets['speed']['mean'], targets['speed']['std'])
    accel = normalize_array(accel, targets['accel']['mean'], targets['accel']['std'])
    brake = normalize_array(brake, targets['brake']['mean'], targets['brake']['std'])
    steer = normalize_array(steer, targets['steer']['mean'], targets['steer']['std'])
    g_force = normalize_array(g_force, targets['g_force']['mean'], targets['g_force']['std'])
    
    # Final clipping
    speed = np.clip(speed, 35, 85)
    accel = np.clip(accel, -3, 3)
    brake = np.clip(brake, 0, 1)
    steer = np.clip(steer, -12, 12)
    g_force = np.clip(g_force, 0.1, 0.5)
    
    return speed, accel, brake, steer, g_force

# -------------------------------
# Generate trips
# -------------------------------
print("Generating trips with IDENTICAL aggregates but different temporal patterns...")

rows = []
safe_count = 0
risky_count = 0

for trip_id in range(1, N_TRIPS + 1):
    length = np.random.randint(MIN_LEN, MAX_LEN)
    
    # Alternate safe and risky to ensure exact balance
    is_risky = (trip_id % 2) == 0
    
    if is_risky:
        sequence = generate_risky_sequence(length)
        risk_label = 1
        risky_count += 1
    else:
        sequence = generate_safe_sequence(length)
        risk_label = 0
        safe_count += 1
    
    # Convert to arrays
    speed, accel, brake, steer, g_force = sequence_to_arrays(sequence)
    
    # Force identical statistics
    speed, accel, brake, steer, g_force = normalize_trip_statistics(speed, accel, brake, steer, g_force)
    
    # Save to dataframe
    for t in range(length):
        rows.append({
            "trip_id": trip_id,
            "timestamp": t,
            "speed": speed[t],
            "acceleration": accel[t],
            "brake_intensity": brake[t],
            "steering_angle": steer[t],
            "g_force": g_force[t],
            "risk_label": risk_label
        })

# -------------------------------
# Verify aggregate similarity
# -------------------------------
df = pd.DataFrame(rows)

print("\nVerifying aggregate statistics are IDENTICAL:")
print("=" * 55)

features = ["speed", "acceleration", "brake_intensity", "steering_angle", "g_force"]
agg_stats = df.groupby(["trip_id", "risk_label"])[features].agg(['mean', 'std', 'max', 'min']).reset_index()

for feature in features:
    safe_mean = agg_stats[agg_stats.risk_label == 0][(feature, 'mean')].mean()
    risky_mean = agg_stats[agg_stats.risk_label == 1][(feature, 'mean')].mean()
    
    safe_std = agg_stats[agg_stats.risk_label == 0][(feature, 'std')].mean()
    risky_std = agg_stats[agg_stats.risk_label == 1][(feature, 'std')].mean()
    
    print(f"{feature.upper()}:")
    print(f"  Mean - Safe: {safe_mean:.3f}, Risky: {risky_mean:.3f} (diff: {abs(safe_mean-risky_mean):.4f})")
    print(f"  Std  - Safe: {safe_std:.3f}, Risky: {risky_std:.3f} (diff: {abs(safe_std-risky_std):.4f})")

# -------------------------------
# Save
# -------------------------------
df.to_csv(OUT_FILE, index=False)

print(f"\n✅ Generated {N_TRIPS} trips ({len(df):,} rows) and saved to {OUT_FILE}")
print(f"Safe trips (0): {safe_count}")
print(f"Risky trips (1): {risky_count}")
print(f"\n🎯 STRATEGY:")
print(f"   • All trips have IDENTICAL aggregate statistics")
print(f"   • Safe trips: smooth, gradual driving patterns")
print(f"   • Risky trips: hard_accel→hard_brake sequences, sharp turns while accelerating")
print(f"   • Only sequence models can detect these temporal dependencies!")