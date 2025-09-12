#!/bin/bash
# Train all ML + AI models

echo "Training all models..."
python src/train_all_models.py

echo "Metrics available in data/model_metrics.csv and data/roc_curves.csv"
