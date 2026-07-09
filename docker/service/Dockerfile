# Train + serve share ONE image (built from the pinned requirements.txt) so the
# mlflow-logged model round-trips byte-for-byte and featuregen is identical on both sides.
# Python 3.12 matches the env that validated walk-forward AUC 0.654.
FROM python:3.12-slim

# libgomp1: OpenMP runtime for scikit-learn / xgboost. curl-free healthcheck uses python.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (see .dockerignore for what's excluded: .venv, mlflow-data, dbs, parquet, ...)
COPY . .

EXPOSE 8100

# Default = the prediction service; the retrain job overrides `command` in compose.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8100"]
