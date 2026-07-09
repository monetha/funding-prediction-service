FROM python:3.11-slim

# MLflow tracking + registry server (sqlite backend, local artifact store).
# Pinned so the registry-alias API and model schema stay reproducible.
ARG MLFLOW_VERSION=2.22.0
RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "sqlalchemy>=2.0" \
        "alembic>=1.13"

# Persisted at runtime via a bind mount / named volume.
WORKDIR /mlflow
EXPOSE 5000

# Bind 0.0.0.0 inside the container; the host publishes it on 127.0.0.1 only.
# --serve-artifacts proxies artifact up/download through the server so clients
# never need direct filesystem access to /mlflow/mlartifacts.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/mlartifacts", \
     "--serve-artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
