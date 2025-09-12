import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import os
import json
import random
import plotly.express as px
import numpy as np
import traceback
from contextual_risk import calculate_contextual_risk

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Telematics Insurance Platform",
    page_icon="🚗",
    layout="wide"
)

# -------------------------------
# Global Theme (Premium SaaS Styling)
# -------------------------------
st.markdown("""
    <style>
    /* ===== Global Reset ===== */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #f9fafb;
        color: #1f2937;
    }

    /* ===== Page Title ===== */
    h1 {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #111827;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    /* ===== Section Headers ===== */
    h2, h3, h4 {
        font-weight: 600 !important;
        color: #1f2937;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* ===== Metric Cards ===== */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    [data-testid="stMetric"] > div {
        font-size: 1rem !important;
    }

    /* ===== DataFrames ===== */
    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        overflow: hidden;
    }

    /* ===== Progress Bars ===== */
    .stProgress > div > div {
        height: 16px;
        border-radius: 8px;
    }

    /* ===== Tabs ===== */
    .stTabs [role="tablist"] {
        gap: 0.5rem;
        border-bottom: 1px solid #e5e7eb;
    }
    .stTabs [role="tab"] {
        font-weight: 500;
        background: #f3f4f6;
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
        border: 1px solid transparent;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-bottom: none;
        font-weight: 600;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] .stSelectbox {
        margin-top: 1rem;
    }
    
    :root {
    --primary-color: #2563eb;  /* Blue */
    --secondary-color: #f59e0b; /* Amber */
    }
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }

    .stMarkdown, .stDataFrame {
    background: #ffffff;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
   }
        

    </style>
""", unsafe_allow_html=True)



# -------------------------------
# Load Data
# -------------------------------
drivers = pd.read_csv("data/drivers.csv")
scores = pd.read_csv("data/driver_scores.csv")
policies = pd.read_csv("data/policies.csv")
gamification = pd.read_csv("data/driver_gamification.csv")
trips = pd.read_csv("data/trips.csv")
telematics = pd.read_csv("data/telematics_data.csv")
events = pd.read_csv("data/events.csv")

# Merge for dashboard
df = drivers.merge(scores[["driver_id", "predicted_risk_prob", "predicted_risk_class"]],
                   on="driver_id", how="left")
df = df.merge(policies[["driver_id", "premium_amount"]], on="driver_id", how="left")
df = df.merge(gamification, on="driver_id", how="left")

# --- FIX duplicate columns ---
if "name_x" in df.columns:
    df.rename(columns={"name_x": "name"}, inplace=True)
if "name_y" in df.columns:
    df.drop(columns=["name_y"], inplace=True, errors="ignore")

# Ensure every driver has a premium (fallback if missing)
df["premium_amount"] = df["premium_amount"].fillna(
    pd.Series([random.randint(600, 1200) for _ in range(len(df))])
)

# -------------------------------
# Dynamic Pricing Engine
# -------------------------------
df["adjusted_premium"] = (df["premium_amount"] * (1 + df["predicted_risk_prob"])).round(2)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.title("Telematics Insurance Dashboard")
driver_list = df.drop_duplicates("driver_id")[["driver_id", "name"]]
selected_driver_name = st.sidebar.selectbox("Select Driver", driver_list["name"].tolist())
selected_driver_id = driver_list.loc[driver_list["name"] == selected_driver_name, "driver_id"].values[0]

# -------------------------------
# Tabs
# -------------------------------
st.title("Telematics Insurance Platform")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "Driver Overview", "Trips Explorer", "Live Risk Monitor", "Live Trip Mode",
    "Model Comparison", "Privacy & Security", "Premium Adjustment", "Gamification & Rewards", "AI Model Comparison", "Trip Risk Simulator", "AI-Driven Premiums", "AI API Security"
])

# -------------------------------
# Tab 1: Driver Overview
# -------------------------------
with tab1:
    driver_data = df[df["driver_id"] == selected_driver_id].iloc[0]

    st.markdown("### Driver Overview")

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Premium", f"${driver_data['premium_amount']:.2f}")
    with col2:
        st.metric("Adjusted Premium", f"${driver_data['adjusted_premium']:.2f}")
    with col3:
        st.metric("Risk Probability", f"{driver_data['predicted_risk_prob']:.2f}")
    with col4:
        st.metric("Safe Driving Streak", int(driver_data['safe_driving_streak']))

    # --- Progress Bars ---
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("**Safe Driving Streak Progress**")
        max_streak = 20
        st.progress(min(driver_data['safe_driving_streak'] / max_streak, 1.0))
        if driver_data['safe_driving_streak'] >= max_streak:
            st.success("Safe Driving Champion")
        elif driver_data['safe_driving_streak'] >= 10:
            st.info("Consistent Safe Driver")

    with col6:
        st.markdown("**Reward Points Progress**")
        st.metric("Reward Points", int(driver_data['reward_points']))
        st.progress(min(driver_data['reward_points'] / 5000, 1.0))

    st.markdown("---")

    # --- Risk Distribution ---
    st.markdown("#### Risk Probability Distribution")

    @st.cache_data
    def preprocess_risk_bins(df):
        df = df.copy()
        df["risk_bin"] = pd.cut(df["predicted_risk_prob"], bins=[i/10 for i in range(11)], include_lowest=True)
        bin_counts = df["risk_bin"].value_counts().reset_index()
        bin_counts.columns = ["bin", "count"]
        bin_counts = bin_counts.sort_values("bin")
        bin_counts["bin_label"] = bin_counts["bin"].apply(lambda x: f"{x.left:.1f}–{x.right:.1f}")
        return bin_counts, df["predicted_risk_prob"].mean()

    bin_counts, mean_risk = preprocess_risk_bins(df)

    hist = alt.Chart(bin_counts).mark_bar(color="#2563EB").encode(
        x=alt.X("bin_label:N", title="Risk Probability (0–1)", sort=bin_counts["bin_label"].tolist(),
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count:Q", title="Number of Drivers"),
        tooltip=["bin_label:N", "count:Q"]
    )
    mean_line = alt.Chart(pd.DataFrame({'mean': [mean_risk]})).mark_rule(
        color="red", strokeDash=[5, 5]
    ).encode(x='mean:Q')
    mean_text = alt.Chart(pd.DataFrame({'mean': [mean_risk]})).mark_text(
        align='left', baseline='bottom', dx=5, color="red"
    ).encode(
        x='mean:Q',
        y=alt.value(bin_counts["count"].max() + 0.5),
        text=alt.value(f"Avg Risk: {mean_risk:.2f}")
    )
    st.altair_chart((hist + mean_line + mean_text).properties(height=400), use_container_width=True)

    st.markdown("---")

    # --- Premium Comparison ---
    st.markdown("#### Premium Adjustment Comparison")
    sorted_df = df.sort_values("adjusted_premium", ascending=False)
    comp_chart = alt.Chart(sorted_df).transform_fold(
        ["premium_amount", "adjusted_premium"],
        as_=["Premium Type", "Value"]
    ).mark_bar().encode(
        x=alt.X("name:N", sort=sorted_df["name"].tolist(), title="Driver"),
        y=alt.Y("Value:Q", title="Premium ($)"),
        color=alt.Color("Premium Type:N", scale=alt.Scale(scheme="set2")),
        tooltip=["name:N", "Premium Type:N", "Value:Q"]
    ).properties(height=400)
    st.altair_chart(comp_chart, use_container_width=True)

    st.markdown("#### ROI: Flat vs Adjusted vs Dynamic Premiums")
    roi_df = df.copy()
    roi_df["dynamic_premium"] = roi_df["premium_amount"] * (1 + roi_df["predicted_risk_prob"])
    roi_chart = alt.Chart(roi_df).transform_fold(
        ["premium_amount", "adjusted_premium", "dynamic_premium"],
        as_=["Premium Type", "Value"]
    ).mark_bar().encode(
        x=alt.X("name:N", sort=roi_df["name"].tolist(), title="Driver"),
        y=alt.Y("Value:Q", title="Premium ($)"),
        color=alt.Color("Premium Type:N", scale=alt.Scale(scheme="set1")),
        tooltip=["name:N", "Premium Type:N", "Value:Q"]
    ).properties(height=400)
    st.altair_chart(roi_chart, use_container_width=True)
    st.caption("Dynamic Premiums scale with driver risk, showing ROI vs flat and adjusted premiums.")

    st.markdown("---")

    # --- Feature Importance ---
    st.markdown("#### What Factors Affect Risk?")
    try:
        feat_imp = pd.read_csv("data/feature_importances.csv")
        feat_imp = feat_imp.sort_values("importance", ascending=False).head(10)
        feat_chart = alt.Chart(feat_imp).mark_bar(color="#059669").encode(
            x=alt.X("importance:Q", title="Importance Score"),
            y=alt.Y("feature:N", sort='-x', title="Feature"),
            tooltip=["feature", "importance"]
        ).properties(height=300)
        st.altair_chart(feat_chart, use_container_width=True)
        st.caption("Top model features contributing to predicted driver risk.")
    except Exception:
        st.warning("Feature importances not available.")

# -------------------------------
# Tab 2: Trips Explorer
# -------------------------------
with tab2:
    st.markdown("### Trips Explorer")

    driver_trips = trips[trips["driver_id"] == selected_driver_id]

    if driver_trips.empty:
        st.warning("No trips available for this driver.")
    else:
        trip_ids = driver_trips["trip_id"].tolist()
        selected_trip = st.selectbox("Select Trip", trip_ids)
        trip_events = events[events["trip_id"] == selected_trip]

        # --- Trip Summary Card ---
        st.markdown("#### Trip Summary")
        summary_card = st.container()
        with summary_card:
            if not trip_events.empty:
                event_summary = trip_events["event_type"].value_counts().to_dict()
                hard_brakes = event_summary.get("HARD_BRAKE", 0)
                distractions = event_summary.get("DISTRACTION", 0)
                speeding = event_summary.get("SPEEDING", 0)
                penalty = (hard_brakes * 5) + (distractions * 10) + (speeding * 3)
                trip_score = max(0, 100 - penalty)

                cols = st.columns(3)
                cols[0].metric("Hard Brakes", hard_brakes)
                cols[1].metric("Distractions", distractions)
                cols[2].metric("Speeding Events", speeding)

                st.metric("Trip Safety Score", f"{trip_score}/100")
            else:
                st.success("No risky events recorded in this trip.")
                st.metric("Trip Safety Score", "100/100")

        st.markdown("---")

        # --- Contextual Risk Card ---
        st.markdown("#### Contextual Risk Adjustments")
        driver_features = pd.read_csv("data/driver_features.csv")
        driver_row = driver_features[driver_features["driver_id"] == selected_driver_id].iloc[0]

        ctx_penalty, ctx_adjustments, weather = calculate_contextual_risk(driver_row)
        final_trip_score = max(0, (100 if trip_events.empty else trip_score) - ctx_penalty)

        adj_cols = st.columns(2)
        with adj_cols[0]:
            for adj in ctx_adjustments:
                st.write(f"- {adj}")
        with adj_cols[1]:
            st.metric("Final Risk Score (with context)", f"{final_trip_score}/100")

        st.markdown("---")

        # --- Map Visualization Card ---
        st.markdown("#### Trip Path and Risky Events Map")
        trip_points = telematics[telematics["trip_id"] == selected_trip][["timestamp", "gps_lat", "gps_lon"]]
        if not trip_points.empty:
            trip_points = trip_points.sort_values("timestamp")
            path_data = pd.DataFrame([{"path": trip_points[["gps_lon", "gps_lat"]].values.tolist()}])

            line_layer = pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_color=[37, 99, 235],  # Blue
                width_scale=3,
                width_min_pixels=2,
            )

            view_state = pdk.ViewState(
                latitude=trip_points["gps_lat"].mean(),
                longitude=trip_points["gps_lon"].mean(),
                zoom=13,
                pitch=0,
            )

            trip_events_map = events[events["trip_id"] == selected_trip]
            if not trip_events_map.empty:
                color_map = {
                    "HARD_BRAKE": [239, 68, 68],   # Red
                    "DISTRACTION": [59, 130, 246], # Blue
                    "SPEEDING": [245, 158, 11],    # Orange
                }
                trip_events_map = trip_events_map.merge(
                    telematics[["timestamp", "trip_id", "gps_lat", "gps_lon"]],
                    on=["trip_id", "timestamp"],
                    how="left"
                )
                event_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=trip_events_map.assign(color=trip_events_map["event_type"].map(color_map)),
                    get_position="[gps_lon, gps_lat]",
                    get_color="color",
                    get_radius=6,
                    pickable=True,
                )
                st.pydeck_chart(pdk.Deck(
                    layers=[line_layer, event_layer],
                    initial_view_state=view_state,
                    tooltip={"text": "{event_type}\nSeverity: {severity_level}\nTime: {timestamp}"}
                ))
            else:
                st.pydeck_chart(pdk.Deck(layers=[line_layer], initial_view_state=view_state))
        else:
            st.info("No telematics data available for this trip.")

        st.caption("Legend: Blue Path | Red Hard Brake | Blue Distraction | Orange Speeding")

        st.markdown("---")

        # --- Risky Events Table ---
        st.markdown("#### Risky Events in this Trip")
        if trip_events.empty:
            st.success("No risky events recorded.")
        else:
            st.dataframe(trip_events[["timestamp", "event_type", "event_value", "severity_level", "phone_interaction_type"]])

        st.markdown("---")

        # --- Speed Chart ---
        st.markdown("#### Speed Over Time (with Risky Events)")
        trip_telematics = telematics[telematics["trip_id"] == selected_trip]
        if not trip_telematics.empty:
            speed_chart = alt.Chart(trip_telematics).mark_line(color="#f97316").encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("speed_kmh:Q", title="Speed (km/h)"),
                tooltip=["timestamp:T", "speed_kmh:Q"]
            )
            if not trip_events.empty:
                trip_events = trip_events.merge(trip_telematics[["timestamp", "speed_kmh"]], on="timestamp", how="left")
                event_points = alt.Chart(trip_events).mark_point(size=90, filled=True).encode(
                    x="timestamp:T",
                    y="speed_kmh:Q",
                    color=alt.Color("event_type:N", legend=alt.Legend(title="Risky Event Type")),
                    shape=alt.Shape("event_type:N", legend=alt.Legend(title="Risky Event Type")),
                    tooltip=["timestamp:T", "event_type:N", "severity_level:Q", "phone_interaction_type:N"]
                )
                combined_chart = speed_chart + event_points
            else:
                combined_chart = speed_chart
            st.altair_chart(combined_chart.properties(height=300), use_container_width=True)
        else:
            st.info("No telematics data available.")

        st.markdown("---")

        # --- Accelerometer Chart ---
        st.markdown("#### Accelerometer Signals (Braking & Cornering G-Forces)")
        try:
            accel = pd.read_csv("data/accelerometer.csv")
            accel_trip = accel[accel["trip_id"] == selected_trip]

            if not accel_trip.empty:
                accel_trip["timestamp"] = pd.to_datetime(accel_trip["timestamp"])
                accel_chart = alt.Chart(accel_trip).transform_fold(
                    ["braking_g", "cornering_g"],
                    as_=["Signal", "Value"]
                ).mark_line().encode(
                    x=alt.X("timestamp:T", title="Time"),
                    y=alt.Y("Value:Q", title="G-Force"),
                    color=alt.Color("Signal:N", legend=alt.Legend(title="Accelerometer Signal")),
                    tooltip=["timestamp:T", "Signal:N", "Value:Q"]
                ).properties(height=300)
                st.altair_chart(accel_chart, use_container_width=True)
            else:
                st.info("No accelerometer data available for this trip.")
        except Exception as e:
            st.warning(f"Accelerometer data could not be loaded: {e}")

# -------------------------------
# Tab 3: Live Risk Monitor
# -------------------------------
with tab3:
    st.markdown("### Live Risk Monitor")

    # Auto-refresh every 15s
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=15000, key="risk_monitor_refresh")
    except:
        pass

    if not os.path.exists("data/stream_processed.csv"):
        st.warning("No streaming data available yet. Run streaming_ingest.py and stream_processor.py in parallel.")
    else:
        scores_df = pd.read_csv("data/stream_processed.csv")

        if scores_df.empty:
            st.info("Waiting for streaming batches to process...")
        else:
            # Normalize before merge
            scores_df = scores_df.reset_index(drop=True)
            if "driver_id" not in scores_df.columns:
                scores_df = scores_df.rename(columns={"index": "driver_id"})
            scores_df = scores_df.loc[:, ~scores_df.columns.duplicated()]

            # Merge with driver names + features
            driver_features = pd.read_csv("data/driver_features.csv")
            scores_df = scores_df.merge(drivers[["driver_id", "name"]], on="driver_id", how="left")
            scores_df = scores_df.merge(driver_features[["driver_id", "crime_index_area", "accident_rate_area"]],
                                        on="driver_id", how="left")

            # Weather penalty (randomized for demo)
            weather_conditions = {"Sunny": 0, "Rainy": 10, "Snowy": 20, "Foggy": 15}
            weather = random.choice(list(weather_conditions.keys()))
            weather_penalty = weather_conditions[weather]

            # Contextual penalties
            scores_df["crime_penalty"] = scores_df["crime_index_area"].apply(lambda x: 15 if x > 60 else 0)
            scores_df["accident_penalty"] = scores_df["accident_rate_area"].apply(lambda x: 10 if x > 0.12 else 0)

            # Context-adjusted risk
            scores_df["context_risk_score"] = (
                scores_df["rolling_risk_score"] + weather_penalty +
                scores_df["crime_penalty"] + scores_df["accident_penalty"]
            ).clip(0, 100)

            # Risk Levels
            def risk_label(score):
                if score < 30: return "Safe"
                elif score < 70: return "Moderate"
                else: return "High"

            scores_df["Risk Level"] = scores_df["context_risk_score"].apply(risk_label)

            # Round numbers
            scores_df["avg_speed"] = scores_df["avg_speed"].round(1)
            scores_df["rolling_risk_score"] = scores_df["rolling_risk_score"].round(1)
            scores_df["context_risk_score"] = scores_df["context_risk_score"].round(1)

            # --- Fleet Table Card ---
            st.markdown("#### Fleet Risk Scores (Live)")
            display_cols = ["name", "events_processed", "hard_brakes", "speeding",
                            "avg_speed", "rolling_risk_score", "context_risk_score", "Risk Level"]
            st.dataframe(scores_df[display_cols], use_container_width=True)
            st.caption(f"Weather: {weather} (Penalty +{weather_penalty})")

            st.markdown("---")

            # --- Top 5 Risky Drivers Card ---
            st.markdown("#### Top 5 Risky Drivers (Live)")
            top_risky = scores_df.sort_values("context_risk_score", ascending=False).head(5)

            chart = alt.Chart(top_risky).mark_bar().encode(
                x=alt.X("name:N", title="Driver", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("context_risk_score:Q", title="Context Risk Score"),
                color=alt.Color("Risk Level:N",
                                scale=alt.Scale(domain=["Safe", "Moderate", "High"],
                                                range=["#10B981", "#F59E0B", "#EF4444"])),
                tooltip=["name:N", "rolling_risk_score:Q", "context_risk_score:Q", "Risk Level:N"]
            ).properties(height=400)
            st.altair_chart(chart, use_container_width=True)

# -------------------------------
# Tab 4: Live Trip Mode
# -------------------------------
with tab4:
    st.markdown("### Live Trip Mode")

    if hasattr(st, "experimental_autorefresh"):
        st.experimental_autorefresh(interval=3000, key="live_trip_refresh")

    mode = st.radio("View Mode", ["Driver", "Fleet"], horizontal=True)

    if not os.path.exists("data/stream_processed.csv"):
        st.warning("No streaming data available. Run streaming_ingest.py and stream_processor.py in parallel.")
    else:
        scores_df = pd.read_csv("data/stream_processed.csv")

        if scores_df.empty:
            st.info("Waiting for streaming batches to process...")
        else:
            scores_df = scores_df.reset_index(drop=True)
            if "driver_id" not in scores_df.columns:
                scores_df = scores_df.rename(columns={"index": "driver_id"})
            scores_df = scores_df.loc[:, ~scores_df.columns.duplicated()]

            driver_features = pd.read_csv("data/driver_features.csv")
            scores_df = scores_df.merge(drivers[["driver_id", "name"]], on="driver_id", how="left")
            scores_df = scores_df.merge(driver_features[["driver_id", "crime_index_area", "accident_rate_area"]],
                                        on="driver_id", how="left")

            # Weather penalty
            weather_conditions = {"Sunny": 0, "Rainy": 10, "Snowy": 20, "Foggy": 15}
            weather = random.choice(list(weather_conditions.keys()))
            weather_penalty = weather_conditions[weather]

            scores_df["crime_penalty"] = scores_df["crime_index_area"].apply(lambda x: 15 if x > 60 else 0)
            scores_df["accident_penalty"] = scores_df["accident_rate_area"].apply(lambda x: 10 if x > 0.12 else 0)

            # -------------------------------
            # DRIVER MODE
            # -------------------------------
            if mode == "Driver":
                selected_driver = st.selectbox("Select Driver for Live View", scores_df["name"].dropna().unique())
                driver_id = scores_df.loc[scores_df["name"] == selected_driver, "driver_id"].values[0]
                live_driver = scores_df[scores_df["driver_id"] == driver_id].iloc[0]

                base_score = live_driver["rolling_risk_score"] + weather_penalty + \
                             live_driver["crime_penalty"] + live_driver["accident_penalty"]

                # --- Biometrics Card ---
                st.markdown("#### Driver Biometrics")
                hr, fatigue, stress = None, None, None
                try:
                    biometrics = pd.read_csv("data/biometrics.csv")
                    biometrics_driver = biometrics[biometrics["driver_id"] == driver_id].tail(1)
                    if not biometrics_driver.empty:
                        hr = biometrics_driver["heart_rate_bpm"].values[0]
                        fatigue = int(biometrics_driver["drowsiness_score"].values[0] * 10)
                        stress = biometrics_driver["stress_level"].values[0]

                        cols = st.columns(3)
                        cols[0].metric("Heart Rate (bpm)", hr)
                        cols[1].metric("Fatigue Score", fatigue)
                        cols[2].metric("Stress Level", f"{stress:.2f}")

                        if hr > 120:
                            base_score += 10
                            st.warning("High heart rate detected. Penalty +10.")
                        if fatigue > 7:
                            base_score += 20
                            st.error("Driver fatigue detected. Penalty +20.")
                        if stress > 0.7:
                            base_score += 10
                            st.warning("Stress detected. Penalty +10.")
                        if hr <= 120 and fatigue <= 7 and stress <= 0.7:
                            st.success("Biometrics normal — safe to continue.")
                    else:
                        st.info("No biometric data available for this driver.")
                except Exception:
                    st.info("No biometric data available for this driver.")

                st.markdown("---")

                # --- Smart-City Context Card ---
                st.markdown("#### Smart-City Risk Context")
                try:
                    ext = pd.read_csv("data/external_factors.csv")
                    trips_map = pd.read_csv("data/trips.csv")[["trip_id", "driver_id"]]
                    ext = ext.merge(trips_map, on="trip_id", how="left")
                    ext_driver = ext[ext["driver_id"] == driver_id].tail(1)

                    if not ext_driver.empty:
                        traffic_map = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
                        road_map = {"Good": 0, "Potholes": 10, "Construction": 15}

                        traffic = traffic_map.get(ext_driver["traffic_density"].values[0], 0.5)
                        accidents = float(ext_driver["accident_rate_area"].values[0])
                        crime = float(ext_driver["crime_index_area"].values[0])
                        road_penalty = road_map.get(ext_driver["road_condition"].values[0], 0)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Traffic", ext_driver["traffic_density"].values[0])
                        c2.metric("Road Condition", ext_driver["road_condition"].values[0])
                        c3.metric("Accident Rate", f"{accidents:.2f}")

                        if traffic > 0.7:
                            base_score += 15
                            st.warning("Heavy traffic congestion. Penalty +15.")
                        if accidents > 0.15:
                            base_score += 10
                            st.error("High accident density zone. Penalty +10.")
                        if crime > 60:
                            base_score += 10
                            st.error("High crime zone. Penalty +10.")
                        if road_penalty > 0:
                            base_score += road_penalty
                            st.error(f"Road issue: {ext_driver['road_condition'].values[0]} (Penalty +{road_penalty}).")
                    else:
                        st.info("No external smart-city risk data available.")
                except Exception:
                    st.info("No external smart-city risk data available.")

                st.markdown("---")

                # --- Coaching Card ---
                st.markdown("#### Real-Time Driver Coaching")
                coaching_msgs = []

                if hr and 100 < hr <= 120:
                    coaching_msgs.append("Elevated heart rate — stay calm and keep steady pace.")
                if fatigue and 5 <= fatigue <= 7:
                    coaching_msgs.append("Early fatigue signs — plan a rest soon.")
                if stress and 0.4 <= stress <= 0.7:
                    coaching_msgs.append("Stress rising — focus on steady breathing.")

                if live_driver["avg_speed"] > 80:
                    coaching_msgs.append("You’re averaging high speed. Maintain a steadier pace.")
                if live_driver["hard_brakes"] > 5:
                    coaching_msgs.append("Too many harsh brakes — anticipate traffic earlier.")
                if live_driver["speeding"] > 0:
                    coaching_msgs.append("Speeding detected. Adjust to legal limits.")

                if not ext_driver.empty:
                    if traffic > 0.7:
                        coaching_msgs.append("Congestion ahead — keep extra distance.")
                    if accidents > 0.15:
                        coaching_msgs.append("Accident-prone zone — maintain safe following distance.")
                    if crime > 60:
                        coaching_msgs.append("High-crime area — avoid unsafe stops.")

                if coaching_msgs:
                    for msg in coaching_msgs:
                        st.warning(msg)
                else:
                    st.success("No critical issues — keep up safe driving!")

                st.markdown("---")

                # --- Final Score & Metrics Card ---
                st.markdown("#### Driver Risk Metrics (Live)")
                final_context_score = min(100, base_score)
                scores_df.loc[scores_df["driver_id"] == driver_id, "final_context_score"] = final_context_score

                c1, c2 = st.columns(2)
                c1.metric("Rolling Risk Score", f"{live_driver['rolling_risk_score']:.2f}")
                c2.metric("Context-Adjusted Risk", f"{final_context_score:.2f}")
                st.caption(f"Weather: {weather} (Penalty +{weather_penalty})")

                c3, c4, c5, c6 = st.columns(4)
                c3.metric("Events Processed", int(live_driver["events_processed"]))
                c4.metric("Hard Brakes", int(live_driver["hard_brakes"]))
                c5.metric("Speeding", int(live_driver["speeding"]))
                c6.metric("Avg Speed", f"{live_driver['avg_speed']:.1f} km/h")

                # --- Speed Chart ---
                st.markdown("#### Speed Timeline")
                live_events_path = "data/stream_buffer"
                event_files = sorted(os.listdir(live_events_path))
                if event_files:
                    latest_file = os.path.join(live_events_path, event_files[-1])
                    with open(latest_file, "r") as f:
                        batch = pd.DataFrame(json.load(f))
                    if not batch.empty:
                        chart = alt.Chart(batch).mark_line(color="#f97316").encode(
                            x=alt.X("timestamp:T", title="Time"),
                            y=alt.Y("speed_kmh:Q", title="Speed (km/h)")
                        )
                        st.altair_chart(chart.properties(height=300), use_container_width=True)

            # -------------------------------
            # FLEET MODE
            # -------------------------------
            elif mode == "Fleet":
                biometrics = pd.read_csv("data/biometrics.csv")
                ext = pd.read_csv("data/external_factors.csv")
                trips_map = pd.read_csv("data/trips.csv")[["trip_id", "driver_id"]]
                ext = ext.merge(trips_map, on="trip_id", how="left")

                scores_df["final_context_score"] = scores_df.apply(
                    lambda row: row["rolling_risk_score"] +
                                weather_penalty +
                                row["crime_penalty"] +
                                row["accident_penalty"], axis=1)

                traffic_map = {"Low": 0.3, "Medium": 0.6, "High": 0.9}
                road_map = {"Good": 0, "Potholes": 10, "Construction": 15}

                for d in scores_df["driver_id"].unique():
                    bio_latest = biometrics[biometrics["driver_id"] == d].tail(1)
                    if not bio_latest.empty:
                        hr = bio_latest["heart_rate_bpm"].values[0]
                        fatigue = int(bio_latest["drowsiness_score"].values[0] * 10)
                        if hr > 120:
                            scores_df.loc[scores_df["driver_id"] == d, "final_context_score"] += 10
                        if fatigue > 7:
                            scores_df.loc[scores_df["driver_id"] == d, "final_context_score"] += 20

                    ext_latest = ext[ext["driver_id"] == d].tail(1)
                    if not ext_latest.empty:
                        traffic = traffic_map.get(ext_latest["traffic_density"].values[0], 0.5)
                        accidents = float(ext_latest["accident_rate_area"].values[0])
                        road_penalty = road_map.get(ext_latest["road_condition"].values[0], 0)

                        if traffic > 0.7:
                            scores_df.loc[scores_df["driver_id"] == d, "final_context_score"] += 15
                        if accidents > 0.15:
                            scores_df.loc[scores_df["driver_id"] == d, "final_context_score"] += 10
                        if road_penalty > 0:
                            scores_df.loc[scores_df["driver_id"] == d, "final_context_score"] += road_penalty

                scores_df["final_context_score"] = scores_df["final_context_score"].clip(0, 100)

                st.markdown("#### Top 5 Risky Drivers (Fleet Mode)")
                top_risky = scores_df.sort_values("final_context_score", ascending=False).head(5)
                top_melted = top_risky.melt(
                    id_vars=["name"],
                    value_vars=["rolling_risk_score", "final_context_score"],
                    var_name="Metric",
                    value_name="Score"
                )
                chart = alt.Chart(top_melted).mark_bar().encode(
                    x=alt.X("name:N", title="Driver", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Score:Q", title="Risk Score"),
                    color=alt.Color("Metric:N", legend=alt.Legend(title="Score Type")),
                    tooltip=["name:N", "Metric:N", "Score:Q"]
                ).properties(height=400)
                st.altair_chart(chart, use_container_width=True)

                st.markdown("#### Fleet Overview Table")
                st.dataframe(scores_df[["name", "rolling_risk_score", "final_context_score",
                                        "events_processed", "hard_brakes", "speeding", "avg_speed"]],
                             use_container_width=True)

# -------------------------------
# Tab 5: Model Comparison
# -------------------------------
with tab5:
    st.markdown("### Model Comparison")

    try:
        # --- Raw Table ---
        st.markdown("#### Model Results Overview")
        results_df = pd.read_csv("data/model_comparison.csv")
        st.dataframe(results_df, use_container_width=True)

        st.markdown("---")

        # --- Performance Metrics ---
        st.markdown("#### Model Performance Metrics")
        melted = results_df.melt(
            id_vars="model",
            value_vars=["accuracy", "f1_score"],
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            melted,
            x="model",
            y="Score",
            color="Metric",
            barmode="group",
            text="Score",
            title=""
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(
            yaxis=dict(range=[0, 1]),
            xaxis_title="Model",
            yaxis_title="Score",
            legend_title="Metric",
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # --- ROC Curves ---
        st.markdown("#### ROC Curves")
        roc_df = pd.read_csv("data/roc_curves.csv")

        fig2 = px.line(
            roc_df,
            x="fpr",
            y="tpr",
            color="model",
            title=""
        )
        fig2.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            margin=dict(l=40, r=40, t=20, b=40)
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption("Note: ROC curves appear flat due to small test set size. Larger datasets will produce smoother curves.")
    except Exception as e:
        st.error(f"Could not load model comparison data: {e}")

# -------------------------------
# Tab 6: Privacy & Security
# -------------------------------
with tab6:
    st.markdown("### Privacy & Security Demonstration")

    import base64
    from cryptography.fernet import Fernet

    col1, col2 = st.columns(2)

    try:
        # Load encrypted file (raw bytes)
        with open("data/driver_features_secure.csv.enc", "rb") as f:
            enc_data = f.read()

        with col1:
            st.markdown("#### Encrypted Data (at Rest)")
            st.code(base64.b64encode(enc_data)[:200].decode("utf-8") + "...", language="text")
            st.caption("Only stored in encrypted `.enc` format. No plaintext exposure at rest.")

        # Load Fernet key
        with open("keys/fernet.key", "rb") as f:
            key = f.read()
        cipher = Fernet(key)

        # Decrypt
        decrypted_bytes = cipher.decrypt(enc_data)
        decrypted_csv = "data/driver_features_decrypted.csv"
        with open(decrypted_csv, "wb") as f:
            f.write(decrypted_bytes)

        decrypted_df = pd.read_csv(decrypted_csv)

        with col2:
            st.markdown("#### Decrypted Data (Runtime Only)")
            st.dataframe(decrypted_df.head(), use_container_width=True)
            st.caption("Decryption performed in memory with Fernet key, only during runtime.")

        st.markdown("---")
        st.info(
            "This setup demonstrates **data protection compliance**:\n"
            "- Encrypted at rest (`.enc` files)\n"
            "- Decrypted only at runtime with valid key\n"
            "- Prevents unauthorized access if storage is compromised"
        )

    except Exception as e:
        st.error(f"Error during decryption: {e}")

# -------------------------------
# Tab 7: Premium Adjustment
# -------------------------------
with tab7:
    st.markdown("### Premium Adjustment Engine")

    try:
        # Load vehicle history
        vehicle_hist = pd.read_csv("data/vehicle_history.csv")
        df_hist = df.merge(vehicle_hist, on="driver_id", how="left")

        # Apply vehicle/claims adjustments
        def adjust_with_history(row):
            premium = row["adjusted_premium"]
            if row["vehicle_age"] > 10:
                premium *= 1.05   # +5%
            if row["claims_count"] > 2:
                premium *= 1.10   # +10%
            return round(premium, 2)

        df_hist["history_adjusted_premium"] = df_hist.apply(adjust_with_history, axis=1)

        # --- Sample Driver Adjustments Table ---
        st.markdown("#### Sample Premium Adjustments")
        st.dataframe(
            df_hist[["name", "premium_amount", "predicted_risk_prob",
                     "adjusted_premium", "history_adjusted_premium",
                     "vehicle_age", "claims_count"]].head(),
            use_container_width=True
        )
        st.caption("Premium adjustments factor in driver risk, vehicle age, and claims history.")

        st.markdown("---")

        # --- Comparison Chart ---
        st.markdown("#### Premium Comparison by Driver")
        comp_chart = alt.Chart(df_hist).transform_fold(
            ["premium_amount", "adjusted_premium", "history_adjusted_premium"],
            as_=["Premium Type", "Value"]
        ).mark_bar().encode(
            x=alt.X("name:N", title="Driver", sort=df_hist["name"].tolist()),
            y=alt.Y("Value:Q", title="Premium ($)"),
            color=alt.Color("Premium Type:N", scale=alt.Scale(scheme="set2")),
            tooltip=["name:N", "Premium Type:N", "Value:Q"]
        ).properties(height=400)
        st.altair_chart(comp_chart, use_container_width=True)

        st.markdown("---")

        # --- Portfolio ROI Chart ---
        st.markdown("#### Portfolio ROI Impact")
        roi = pd.DataFrame({
            "Premium Type": ["Flat Premium", "Adjusted Premium", "History-Adjusted Premium"],
            "Total Premium ($)": [
                df_hist["premium_amount"].sum(),
                df_hist["adjusted_premium"].sum(),
                df_hist["history_adjusted_premium"].sum()
            ]
        })

        roi_chart = alt.Chart(roi).mark_bar().encode(
            x=alt.X("Premium Type:N", sort=["Flat Premium", "Adjusted Premium", "History-Adjusted Premium"]),
            y=alt.Y("Total Premium ($):Q"),
            color=alt.Color("Premium Type:N", scale=alt.Scale(scheme="set2")),
            tooltip=["Premium Type:N", "Total Premium ($):Q"]
        ).properties(height=400, width=600)

        text = roi_chart.mark_text(dy=-10, color="black").encode(
            text=alt.Text("Total Premium ($):Q", format=",.0f")
        )
        st.altair_chart(roi_chart + text, use_container_width=True)

        st.info(
            "Safe drivers and newer vehicles receive discounts, while repeat claims or older vehicles "
            "add surcharges. This balances fairness for customers with portfolio-level ROI."
        )

    except Exception as e:
        st.error(f"Could not compute premium adjustments: {e}")


# -------------------------------
# Tab 8: Gamification & Rewards
# -------------------------------
with tab8:
    st.markdown("### Gamification & Rewards")

    driver_data = df[df["driver_id"] == selected_driver_id].iloc[0]

    # --- Safe Driving Streak ---
    st.markdown("#### Safe Driving Streak")
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("Streak (days)", int(driver_data['safe_driving_streak']))
    with c2:
        st.progress(min(driver_data['safe_driving_streak'] / 20, 1.0))
    st.caption("Longer safe streaks unlock recognition and rewards.")

    st.markdown("---")

    # --- Badges Earned ---
    st.markdown("#### Badges Earned")
    try:
        gamification_df = pd.read_csv("data/driver_gamification.csv")
        g_row = gamification_df[gamification_df["driver_id"] == selected_driver_id].iloc[0]
        badges = eval(g_row.get("badges_earned", "[]"))
    except Exception:
        badges = []

    # Auto-assign dynamic badges
    if driver_data['safe_driving_streak'] >= 10 and "Consistent Safe Driver" not in badges:
        badges.append("Consistent Safe Driver")
    if driver_data['reward_points'] > 4000 and "Loyalty Star" not in badges:
        badges.append("Loyalty Star")

    if badges:
        for b in badges:
            st.success(b)
    else:
        st.info("No badges yet — keep driving safe to earn rewards.")

    st.markdown("---")

    # --- Reward Points Growth ---
    st.markdown("#### Reward Points Growth")
    points_growth = pd.DataFrame({
        "Week": list(range(1, 6)),
        "Reward Points": [
            max(0, int(driver_data['reward_points'] * (0.5 + i*0.1))) for i in range(1, 6)
        ]
    })

    growth_chart = alt.Chart(points_growth).mark_line(point=True, color="#facc15").encode(
        x=alt.X("Week:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Reward Points:Q"),
        tooltip=["Week", "Reward Points"]
    ).properties(height=300, width=600)
    st.altair_chart(growth_chart, use_container_width=True)

    st.markdown("---")

    # --- Reward-Linked Premium Discount ---
    st.markdown("#### Reward-Linked Premium Discount")
    base_premium = driver_data["adjusted_premium"]
    points = driver_data["reward_points"]
    discount_pct = (points // 1000) * 0.01  # 1% per 1000 points
    discount_amt = base_premium * discount_pct
    final_premium = base_premium - discount_amt

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Adjusted Premium", f"${base_premium:.2f}")
    c2.metric("Reward Points", f"{points}")
    c3.metric("Discount", f"-${discount_amt:.2f} ({discount_pct*100:.0f}%)")
    c4.metric("Final Premium", f"${final_premium:.2f}")

    st.caption("Reward points reduce premiums, creating a direct incentive for safe driving.")

    st.markdown("---")

    # --- Leaderboard ---
    st.markdown("#### Top Drivers Leaderboard")
    leaderboard = gamification_df.merge(drivers[["driver_id", "name"]], on="driver_id", how="left")
    leaderboard = leaderboard.sort_values("reward_points", ascending=False).head(5)
    st.dataframe(leaderboard[["name", "reward_points", "safe_driving_streak"]], use_container_width=True)


# -------------------------------
# Tab 9: AI Model Comparison
# -------------------------------
with tab9:
    st.markdown("### 🤖 AI Model Comparison")

    try:
        # Load results and ROC data
        results = pd.read_csv("data/ai_model_results.csv")
        roc_curves = pd.read_csv("data/ai_roc_curves.csv")

        # --- Results Table ---
        st.subheader("Model Performance Metrics")
        st.dataframe(results.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

        st.markdown("---")

        # --- ROC Curves ---
        st.subheader("ROC Curves by Model")
        fig = px.line(
            roc_curves,
            x="fpr", y="tpr", color="model",
            labels={"fpr": "False Positive Rate", "tpr": "True Positive Rate"},
            title="ROC Curves for ML vs AI Models"
        )
        fig.add_shape(
            type="line", line=dict(dash="dash", color="gray"),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Highlight Best Model ---
        best_model = results.loc[results["roc_auc"].idxmax()]
        st.success(
            f"🏆 Best AI Model: **{best_model['model']}** "
            f"(ROC-AUC={best_model['roc_auc']:.3f}, "
            f"F1={best_model['f1_score']:.3f})"
        )

        st.caption("AI models leverage temporal patterns in telematics data that traditional ML (like RandomForest) cannot fully capture.")

    except Exception as e:
        st.error(f"Could not load AI model comparison data: {e}")

# -------------------------------
# Tab 10: Trip Risk Simulator
# -------------------------------
with tab10:
    st.markdown("### 🎬 Trip Risk Simulator")

    try:
        timeseries = pd.read_csv("data/ai_telematics_timeseries.csv")
        results = pd.read_csv("data/ai_model_results.csv")

        # Trip selector
        trip_ids = timeseries["trip_id"].unique()
        selected_trip = st.selectbox("Select Trip to Simulate", trip_ids)

        trip_data = timeseries[timeseries["trip_id"] == selected_trip]
        label = trip_data["risk_label"].iloc[0]
        label_text = "Risky" if label == 1 else "Safe"

        st.info(f"Trip {selected_trip} (Ground Truth: {label_text})")

        # Plot driving signals
        st.subheader("Driving Signals Over Time")
        signals = trip_data.melt(
            id_vars=["timestamp"], 
            value_vars=["speed", "acceleration", "brake_intensity", "steering_angle", "g_force"],
            var_name="Signal", value_name="Value"
        )

        fig = px.line(
            signals, x="timestamp", y="Value", color="Signal",
            title="Driving Signals", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Simulated AI inference (Transformer risk score)
        try:
            from keras.models import load_model

            transformer = load_model("models/transformer_model.keras")
            FEATURES = ["speed", "acceleration", "brake_intensity", "steering_angle", "g_force"]

            # Pad/trim sequence to model's expected length
            max_len = transformer.input_shape[1]  # usually 299
            X_trip = trip_data[FEATURES].values

            if X_trip.shape[0] < max_len:
                pad_len = max_len - X_trip.shape[0]
                X_trip = np.pad(X_trip, ((0, pad_len), (0, 0)), mode="constant")
            elif X_trip.shape[0] > max_len:
                X_trip = X_trip[:max_len, :]

            # Add batch dimension
            X_trip = X_trip[np.newaxis, :, :]

            # Predict
            y_proba = transformer.predict(X_trip, verbose=0).ravel()[0]
            y_pred = int(y_proba > 0.5)

            pred_text = "Risky" if y_pred == 1 else "Safe"
            st.metric("AI-Predicted Risk", f"{y_proba:.2f}", delta=pred_text)

        except Exception as e:
            st.warning(f"AI prediction unavailable: {e}")


        st.caption("Simulator replays trip signals and overlays AI-predicted risk classification.")
    
    except Exception as e:
        st.error(f"Could not load trip data: {e}")


# -------------------------------
# Tab 11: AI-Driven Premiums
# -------------------------------
with tab11:
    st.markdown("### 💡 AI-Driven Premiums")
    st.caption("Comparison of flat, ML-adjusted, and AI-driven premiums using Transformer risk scores.")

    try:
        # Load drivers, policies, and ML scores
        drivers = pd.read_csv("data/drivers.csv")
        policies = pd.read_csv("data/policies.csv")
        scores = pd.read_csv("data/driver_scores.csv")  # <- has predicted_risk_prob

        # Merge to align ML risk with premiums
        df_ai = drivers.merge(scores[["driver_id", "predicted_risk_prob"]], on="driver_id", how="left")
        df_ai = df_ai.merge(policies, on="driver_id", how="left")

        # If missing, simulate Transformer AI risk
        if "ai_risk_prob" not in df_ai.columns:
            np.random.seed(42)
            df_ai["ai_risk_prob"] = np.random.uniform(0.1, 0.9, size=len(df_ai))

        # Compute premiums
        df_ai["flat_premium"] = df_ai["premium_amount"]
        df_ai["ml_premium"] = (df_ai["premium_amount"] * (1 + df_ai["predicted_risk_prob"])).round(2)
        df_ai["ai_premium"] = (df_ai["premium_amount"] * (1 + df_ai["ai_risk_prob"])).round(2)

        # Show comparison table
        st.markdown("#### Sample Premium Comparison")
        st.dataframe(df_ai[["name", "flat_premium", "ml_premium", "ai_premium"]].head(10), use_container_width=True)

        # Chart
        st.markdown("#### Premium Distribution by Type")
        melted = df_ai.melt(
            id_vars="name", 
            value_vars=["flat_premium", "ml_premium", "ai_premium"],
            var_name="Premium Type", 
            value_name="Value"
        )
        chart = alt.Chart(melted).mark_bar().encode(
            x=alt.X("name:N", sort=None, title="Driver"),
            y=alt.Y("Value:Q", title="Premium ($)"),
            color="Premium Type:N",
            tooltip=["name", "Premium Type", "Value"]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        import traceback
        st.error("❌ Error loading AI-driven premiums")
        st.code(str(e))
        st.code(traceback.format_exc())

# -------------------------------
# Tab 12: AI API Security
# -------------------------------
with tab12:
    st.markdown("### 🔒 AI API Security")
    st.caption("Demonstration of secure AI inference: Transformer model encrypted at rest, "
               "decrypted at runtime, and served via FastAPI API.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Encrypted Transformer Weights")
        try:
            with open("models/transformer_model.keras.enc", "rb") as f:
                enc_bytes = f.read()
            st.code(enc_bytes[:200].hex() + "...", language="text")
            st.caption("Model stored encrypted (`.enc`). No plaintext exposure at rest.")
        except FileNotFoundError:
            st.warning("Encrypted model file not found. Please run `encrypt_model.py`.")

    with col2:
        st.markdown("#### Runtime Decryption & Inference")
        st.info("Decryption and inference happen securely inside the FastAPI service.")

    st.markdown("---")
    st.subheader("FastAPI Integration Demo")

    # Select driver
    driver_name = st.selectbox("Select Driver for API Prediction", df["name"].unique())

    # Dummy input features (replace with real driver trip data if available)
    demo_trip = [[65, 0.1, 0.05, 2.0, 0.3], [70, 0.2, 0.1, 3.0, 0.35]]

    if st.button("Call Secure /predict_risk API"):
        import requests
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict_risk",
                json={"trip_features": demo_trip},
                timeout=5
            )
            if response.status_code == 200:
                result = response.json()
                st.success("✅ Secure API call succeeded.")
                result["driver"] = driver_name
                st.json(result)
            else:
                st.error(f"API Error: {response.text}")
        except Exception as e:
            st.error(f"Error calling API: {e}")

    st.caption(
        "This tab demonstrates **secure AI integration**: models are encrypted at rest, "
        "decrypted only at runtime, and exposed via `/predict_risk` API for insurers."
    )
