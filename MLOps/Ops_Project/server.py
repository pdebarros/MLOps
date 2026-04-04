"""
Local FastAPI service: KG pipeline (`predict.py`), health checks for Neo4j + GCP (ADC).

Run (from Ops_Project, with .env and ADC on the host):

  uvicorn server:app --reload --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
_KG = _ROOT / "KG_agent"
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config, google_cloud_project  # noqa: E402
from predict import run_kg_pipeline  # noqa: E402

app = FastAPI(title="KG Pipeline API", version="0.1.0")


class PredictRequest(BaseModel):
    user_id: str = Field(..., description="GCS prefix, e.g. u_123", examples=["u_123"])


@app.get("/root")
async def root_welcome() -> dict[str, str]:
    return {
        "message": "Welcome to the Knowledge Graph pipeline API. "
        "Use POST /predict with {\"user_id\": \"...\"} to run the pipeline, "
        "GET /health for Neo4j and GCP checks.",
    }


@app.post("/predict")
async def predict_endpoint(body: PredictRequest) -> dict[str, Any]:
    try:
        return await run_kg_pipeline(body.user_id)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _check_neo4j_sync() -> tuple[bool, str | None]:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )
    try:
        driver.verify_connectivity()
        with driver.session() as session:
            session.run("RETURN 1 AS ok")
        return True, None
    except Exception as e:
        return False, str(e)
    finally:
        driver.close()


def _check_gcs_and_vertex_sync() -> dict[str, Any]:
    """Verify ADC, GCS bucket access, and Vertex AI init."""
    import google.auth
    import vertexai
    from google.cloud import storage

    out: dict[str, Any] = {
        "adc": {"ok": False, "project": None, "error": None},
        "gcs": {"ok": False, "bucket": Config.GCS_BUCKET_NAME, "error": None},
        "vertex_ai": {"ok": False, "project": None, "location": None, "error": None},
    }

    try:
        _credentials, project = google.auth.default()
        pid = project or google_cloud_project()
        out["adc"]["ok"] = True
        out["adc"]["project"] = pid
    except Exception as e:
        out["adc"]["error"] = str(e)
        return out

    try:
        client = storage.Client()
        bucket = client.bucket(Config.GCS_BUCKET_NAME)
        bucket.exists()
        out["gcs"]["ok"] = True
    except Exception as e:
        out["gcs"]["error"] = str(e)

    try:
        loc = (Config.VERTEX_LOCATION or "us-central1").strip() or "us-central1"
        proj = pid or google_cloud_project()
        if not proj:
            out["vertex_ai"]["error"] = "No GCP project id (set GOOGLE_CLOUD_PROJECT or ADC project)."
        else:
            vertexai.init(project=proj, location=loc)
            out["vertex_ai"]["ok"] = True
            out["vertex_ai"]["project"] = proj
            out["vertex_ai"]["location"] = loc
    except Exception as e:
        out["vertex_ai"]["error"] = str(e)

    return out


@app.get("/health")
async def health() -> dict[str, Any]:
    neo4j_ok, neo4j_err = await asyncio.to_thread(_check_neo4j_sync)
    gcp = await asyncio.to_thread(_check_gcs_and_vertex_sync)

    overall = (
        neo4j_ok
        and gcp.get("adc", {}).get("ok")
        and gcp.get("gcs", {}).get("ok")
        and gcp.get("vertex_ai", {}).get("ok")
    )

    return {
        "status": "healthy" if overall else "degraded",
        "neo4j": {
            "ok": neo4j_ok,
            "uri": Config.NEO4J_URI,
            "error": neo4j_err,
        },
        "gcp": gcp,
    }
