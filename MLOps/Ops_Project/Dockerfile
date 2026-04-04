# Local-only API: uses Application Default Credentials from the host (mount gcloud config)
# and Neo4j on the host or another container (see NEO4J_URI).
#
# Build (from Ops_Project):
#   docker build -t kg-api:local .
#
# Run (Mac/Windows/Linux — Neo4j on host, default bolt port):
#   docker run --rm -p 8080:8080 \
#     -v ~/.config/gcloud:/root/.config/gcloud:ro \
#     -v "$(pwd)/KG_agent/.env:/app/KG_agent/.env:ro" \
#     -e NEO4J_URI=bolt://host.docker.internal:7687 \
#     kg-api:local
#
# Linux: if host.docker.internal is missing, add:
#   --add-host=host.docker.internal:host-gateway
#
# ADC: run `gcloud auth application-default login` on the host before `docker run`.

FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
