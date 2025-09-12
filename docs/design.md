# System Design Notes

## Overview
This system is a proof-of-concept (POC) for integrating real-time telematics data into auto insurance pricing. It demonstrates how raw driving signals, contextual data, and biometrics can be securely ingested, processed, and used to calculate fairer, AI-driven premiums.

## Core Design Choices
- **Streaming-first architecture**: Ingestion (`streaming_ingest.py`) and processing (`stream_processor.py`) simulate near real-time driver scoring.
- **Storage**: For POC, CSVs are used. In production, this would scale to S3 or BigQuery for cloud-native storage.
- **Modeling**: Both machine learning (RF, LR, XGBoost, NN) and AI deep learning models (LSTM, GRU, CNN, Transformer, Autoencoder) were tested.
- **Security-first design**: All sensitive driver features are encrypted at rest (`.enc` files) with runtime decryption.
- **Configurability**: Premium calculation engine is modular, allowing insurers to adjust rules for PAYD vs PHYD.

## Scalability
- **Horizontal scaling**: Replace CSVs with Kafka/S3 → stream processing with Spark/Flink.
- **Model serving**: Deploy AI models on TensorFlow Serving or SageMaker.
- **Microservices**: FastAPI endpoints wrap the Transformer model for secure API calls.

