FROM python:3.12-slim

# Kept in its own image: mlflow (full) pins pandas<3, which conflicts with
# this project's pandas==3.0.5 pin in requirements/requirements.txt. The dev
# image logs runs with mlflow-skinny (no pandas dependency) over HTTP against
# the tracking server this image runs; it never touches the store directly.
RUN pip install --no-cache-dir mlflow==3.15.1

EXPOSE 5000

# No --default-artifact-root: leaving it unset keeps the server's default
# proxied ("mlflow-artifacts:/") artifact store, so clients upload/download
# over the same HTTP connection instead of touching the storage path
# directly; required here since the dev container (the client) doesn't
# mount /mlruns at all.
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlruns/mlflow.db", \
     "--artifacts-destination", "/mlruns/mlartifacts", \
     "--host", "0.0.0.0", "--port", "5000", \
     "--allowed-hosts", "mlflow,mlflow:5000,localhost,localhost:5000,127.0.0.1,127.0.0.1:5000"]
