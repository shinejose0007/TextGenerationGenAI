#!/usr/bin/env bash
set -e
echo "1) Running ETL..."
python etl/etl.py
echo "2) Training (this may be slow on CPU; consider lowering epochs/batch size)..."
python training/train.py
echo "3) Evaluating..."
python training/evaluate.py
echo "4) Computing baseline embeddings..."
python monitoring/drift.py
echo "Done. You can run the FastAPI service: cd service && uvicorn app:app --host 0.0.0.0 --port 8000"