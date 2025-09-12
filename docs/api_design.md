# API Design (FastAPI Secure Inference)

## Endpoint: `/predict_risk`
- **Method**: POST
- **Content-Type**: `application/json`

### Request Example
```json
{
  "trip_features": [
    [65, 0.1, 0.05, 2.0, 0.3],
    [70, 0.2, 0.1, 3.0, 0.35]
  ]
}

### Response Example
```json
{
  "risk_score": 0.9658,
  "risk_label": "High"
}

## Security

- Transformer model stored encrypted at rest (.keras.enc).

- Decrypted only in memory at runtime by FastAPI.

- Keys managed in /models/encryption.key.