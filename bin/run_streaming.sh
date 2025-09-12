#!/bin/bash
# Run streaming pipeline (two terminals recommended)

echo "Starting telematics event ingestion..."
python src/streaming_ingest.py &
INGEST_PID=$!

echo "Starting stream processor..."
python src/stream_processor.py &
PROC_PID=$!

wait $INGEST_PID $PROC_PID
