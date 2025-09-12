# Security & Privacy Design

## Principles
Insurance requires strong compliance with data privacy regulations (GDPR, NAIC). This POC demonstrates encryption, decryption, and secure API exposure.

## Measures Implemented
1. **Encryption at Rest**
   - Driver features stored as `.enc` files (`driver_features_secure.csv.enc`).
   - AES (Fernet) encryption with keys stored in `/keys/fernet.key`.

2. **Decryption at Runtime**
   - Features only decrypted when loaded into memory.
   - Prevents unauthorized access to raw sensitive data.

3. **Secure AI API**
   - Transformer model encrypted (`transformer_model.keras.enc`).
   - Decrypted only at runtime by `ai_api.py`.
   - Served via `/predict_risk` FastAPI endpoint.

4. **Compliance**
   - Data minimization: only essential features (speed, braking, biometrics).
   - Modular key rotation and storage for regulatory alignment.

