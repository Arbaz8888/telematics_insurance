from graphviz import Digraph

def create_updated_architecture_diagram(output_path="docs/architecture_diagram_updated"):
    dot = Digraph("TelematicsInsurance", format="png")

    # Global style
    dot.attr(rankdir="LR", fontsize="11")
    dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica")

    # ----------- Data Sources -----------
    dot.node("devices", "Telematics Devices\n(GPS, Accelerometer, Smartphone)", fillcolor="#e3f2fd", color="#1e88e5")
    dot.node("biometrics", "Biometric Sensors\n(Heart Rate, Fatigue, Stress)", fillcolor="#fce4ec", color="#c2185b")
    dot.node("smartcity", "Smart City & External Data\n(Traffic, Road, Weather, Crime)", fillcolor="#e0f2f1", color="#00796b")

    # ----------- Ingestion & Processing -----------
    dot.node("streaming", "Streaming Ingestion\n(streaming_ingest.py)", fillcolor="#ede7f6", color="#5e35b1")
    dot.node("processor", "Stream Processor\n(stream_processor.py)", fillcolor="#f3e5f5", color="#8e24aa")

    # ----------- Storage Layer -----------
    dot.node("storage", "Data Lake / Storage\n(CSVs, Cloud Buckets)", fillcolor="#fff3e0", color="#f57c00")

    # ----------- Security Layer -----------
    dot.node("security", "Privacy & Security\n(Fernet Encryption / Decryption)", fillcolor="#e1f5fe", color="#0277bd")

    # ----------- Risk Modeling -----------
    dot.node("models", "Risk Scoring Models", fillcolor="#e8f5e9", color="#388e3c")

    classic_models = {
        "rf": "RandomForest",
        "lr": "Logistic Regression",
        "xgb": "XGBoost",
        "nn": "Neural Network"
    }
    ai_models = {
        "lstm": "LSTM",
        "gru": "GRU",
        "cnn": "CNN",
        "ae": "Autoencoder",
        "trans": "Transformer"
    }

    for k, v in classic_models.items():
        dot.node(k, v, fillcolor="#c8e6c9", color="#2e7d32")
    for k, v in ai_models.items():
        dot.node(k, v, fillcolor="#fff9c4", color="#fbc02d")

    # ----------- Pricing Engine -----------
    dot.node("pricing", "Dynamic Pricing Engine\n(Flat, ML, AI Premiums)", fillcolor="#fffde7", color="#fbc02d")

    # ----------- APIs -----------
    dot.node("apis", "Secure APIs\n(/predict, /premium)\nFastAPI + Runtime Decryption", fillcolor="#e0f7fa", color="#00838f")

    # ----------- Dashboard Tabs -----------
    dot.node("dashboard", "Streamlit Dashboard\n(12 Tabs)", fillcolor="#fce4ec", color="#d81b60")

    tabs = [
        "Driver Overview", "Trips Explorer", "Live Risk Monitor", "Live Trip Mode",
        "Model Comparison", "Privacy & Security", "Premium Adjustment", "Gamification & Rewards",
        "AI Model Comparison", "Trip Risk Simulator", "AI-Driven Premiums", "AI API Security"
    ]
    for i, t in enumerate(tabs, 1):
        dot.node(f"tab{i}", f"Tab {i}: {t}", fillcolor="#f8bbd0", color="#ad1457")

    # ----------- Edges -----------
    # Sources → Ingestion
    dot.edge("devices", "streaming", label="raw telematics")
    dot.edge("biometrics", "streaming", label="driver biometrics")
    dot.edge("smartcity", "processor", label="contextual risk factors")

    # Ingestion → Processing → Storage
    dot.edge("streaming", "processor", label="event batches")
    dot.edge("processor", "storage", label="cleaned datasets")

    # Security
    dot.edge("security", "storage", label="encrypt at rest")
    dot.edge("security", "dashboard", label="runtime decryption")

    # Storage → Models
    dot.edge("storage", "models", label="training data")

    # Models → Submodels
    for m in classic_models:
        dot.edge("models", m)
    for m in ai_models:
        dot.edge("models", m)

    # Submodels → Pricing Engine
    for m in classic_models:
        dot.edge(m, "pricing", label="risk score")
    for m in ai_models:
        dot.edge(m, "pricing", label="risk score")

    # Pricing Engine → APIs + Dashboard
    dot.edge("pricing", "apis", label="premium calculations")
    dot.edge("pricing", "dashboard", label="premium adjustments")

    # APIs → Dashboard
    dot.edge("apis", "dashboard", label="real-time scoring")

    # Dashboard → Tabs
    for i in range(1, len(tabs) + 1):
        dot.edge("dashboard", f"tab{i}")

    # Render
    dot.render(output_path, format="png", cleanup=True)
    print(f"✅ Updated architecture diagram saved at {output_path}.png")

if __name__ == "__main__":
    create_updated_architecture_diagram()
