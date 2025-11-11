FROM python:3.10-slim
WORKDIR /app
COPY ../training/outputs/t5-small-experiment /app/model || true
COPY app.py /app
RUN pip install --no-cache-dir fastapi uvicorn transformers torch prometheus_client
EXPOSE 8000 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]