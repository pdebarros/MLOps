# Cloud Run Job image for the KG pipeline (experience_pipeline edition).
# Triggered by gitai-upload-api after a user uploads Python files.
#
# Build from Ops_Project/KG_agent/:
#   docker build -t gcr.io/PROJECT/kg-pipeline-job .
#
# All runtime secrets (NEO4J_URI, NEO4J_PASSWORD, etc.) must be set as
# environment variables in the Cloud Run Job definition (or via Secret Manager).
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY kg_requirements.txt .
RUN pip install --no-cache-dir -r kg_requirements.txt

# Copy source files needed by experience_pipeline.py (and the prod modules it imports).
COPY experience_pipeline.py .
COPY prod_pipeline.py .
COPY pipeline2.py .
COPY tools2.py .
COPY config.py .
COPY __init__.py .

ENV PYTHONUNBUFFERED=1
# Ensure the working directory is on sys.path so local imports (config, tools2, pipeline2) resolve.
ENV PYTHONPATH=/app

# --user-id (and optional --neo4j-database) are injected at execution time by
# gitai-upload-api via container arg overrides; experience_pipeline.py accepts
# the same CLI shape as prod_pipeline.py so the API contract is unchanged.
ENTRYPOINT ["python", "experience_pipeline.py"]
