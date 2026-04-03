"""
Knowledge-graph creation only (no MLflow, no eval agent).

Expose **`async def run_kg_pipeline(user_id: str)`** for FastAPI:

    from predict import run_kg_pipeline

    @app.post("/predict")
    async def predict_endpoint(body: PredictBody):
        result = await run_kg_pipeline(body.user_id)
        return result

`run_kg_pipeline` delegates to `KG_agent.pipeline2.run_pipeline` and returns a JSON-serializable
dict (status, user_id, py_file_count, kg_result, etc.).

CLI (optional):

  cd Ops_Project
  python predict.py --user-id u_123
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_KG = _ROOT / "KG_agent"
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

import pipeline2 as kg_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("predict")


async def run_kg_pipeline(user_id: str) -> dict[str, Any]:
    """
    Run summarize + KG build for ``user_id`` (GCS prefix under ``GCS_BUCKET_NAME``).

    Safe to call from an async FastAPI route with ``await``. Returns a plain dict suitable
    for ``JSONResponse`` / ``return result``.

    Raises:
        ValueError: if ``user_id`` is empty or whitespace-only (from ``pipeline2``).
    """
    logger.info("run_kg_pipeline starting for user_id=%r", user_id)
    out = await kg_pipeline.run_pipeline(user_id)
    if out is None:
        return {
            "status": "error",
            "reason": "pipeline_returned_none",
            "user_id": user_id.strip().strip("/"),
        }
    return out


def run_kg_pipeline_sync(user_id: str) -> dict[str, Any]:
    """Sync wrapper for non-async contexts (e.g. scripts, Celery). Uses ``asyncio.run``."""
    return asyncio.run(run_kg_pipeline(user_id))


def main() -> None:
    p = argparse.ArgumentParser(description="Run KG pipeline for one user_id (no MLflow).")
    p.add_argument("--user-id", required=True, help="GCS prefix, e.g. u_123")
    args = p.parse_args()
    result = asyncio.run(run_kg_pipeline(args.user_id))
    print(result)


__all__ = ["run_kg_pipeline", "run_kg_pipeline_sync"]


if __name__ == "__main__":
    main()
