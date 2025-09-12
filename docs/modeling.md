# Modeling Decisions & Results

## Models Tested
- **RandomForest**
- **Logistic Regression**
- **XGBoost**
- **Neural Network (MLP)**
- **LSTM** (sequence-based)
- **GRU** (sequence-based)
- **CNN** (sequence-based)
- **Autoencoder** (unsupervised anomaly detection)
- **Transformer** (state-of-the-art sequence modeling)

## Key Observations
- Traditional ML (RF, XGB) achieved strong baseline performance on tabular features.
- Deep learning models (LSTM, GRU, CNN) captured sequential driving patterns but were less stable without sufficient sequence length.
- **Transformer outperformed all models**, achieving near-perfect classification with ROC-AUC ≈ 1.0 and stable PR-AUC across balanced datasets.

## Why Transformer Wins
- Handles variable-length sequences with attention.
- Learns complex temporal interactions in driving behavior.
- Scales better than RNNs for large datasets.

