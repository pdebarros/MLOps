# Upload API — small image, no KG pipeline deps.
# KG pipeline runs as a separate Cloud Run Job (see Ops_Project/KG_agent/Dockerfile).
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
