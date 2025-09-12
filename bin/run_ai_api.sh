#!/bin/bash
# Launch the secure FastAPI service for AI inference

echo "Starting AI Risk Prediction API on port 8000..."
uvicorn src.ai_api:app --reload --port 8000
