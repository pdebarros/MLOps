"""
MLflow experiment: build a knowledge graph (KG_agent pipeline2) for a user_id, then run the
eval_agent (ADK) to score the graph and persist scores to GCS. Logs Neo4j node/edge counts,
pipeline output, eval transcript, and structured eval metrics from the scoring file.

Run from Ops_Project (repo root for imports):

  cd Ops_Project
  python experiment.py --user-id u_123

Requires: same .env as KG_agent / eval_agent (Neo4j, GCS, Vertex, MLflow tracking URI optional).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Any

# KG_agent uses local `config` import
_ROOT = Path(__file__).resolve().parent
_KG = _ROOT / "KG_agent"
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import mlflow  # noqa: E402
import pipeline2 as kg_pipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("experiment")


async def _final_text_from_events(events_source) -> str | None:
    """Last non-user final response text from an ADK runner stream."""
    last: str | None = None
    async for event in events_source:
        if not event.is_final_response():
            continue
        if getattr(event, "author", None) == "user":
            continue
        if event.content and event.content.parts:
            chunk = "\n".join(
                p.text for p in event.content.parts if getattr(p, "text", None)
            )
            if chunk:
                last = chunk
    return last


async def run_eval_agent_session(user_id: str) -> str:
    """Single ADK session: full eval workflow including save_scores_to_gcs."""
    from google.adk.runners import Runner
    from google.adk.sessions.in_memory_session_service import InMemorySessionService
    from google.genai import types

    from eval_agent.agent import root_agent

    runner = Runner(
        app_name="kg-experiment-eval",
        agent=root_agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    session_id = f"exp-{uuid.uuid4().hex}"
    adk_user = f"experiment-{user_id}"
    prompt = (
        f"My user_id is `{user_id}`. Perform the complete knowledge-graph evaluation: "
        "sample several neighborhoods (list_entity_id_samples / fetch_neighborhood_from_neo4j), "
        "ground with retrieve_from_rag_corpus, assign per-sample scores, aggregate averages, "
        "output the ## Final scores block, then call save_scores_to_gcs with user_id, "
        "per_sample_scores_json, and all required fields."
    )
    msg = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )
    text = await _final_text_from_events(
        runner.run_async(
            user_id=adk_user,
            session_id=session_id,
            new_message=msg,
        )
    )
    return text or ""


def neo4j_counts_for_user_prefix(user_id: str) -> tuple[int, int]:
    """Count :Entity and :REL where entity ids start with ``<user_id>/``."""
    from config import Config
    from neo4j import GraphDatabase

    uid = kg_pipeline.normalize_user_id(user_id)
    prefix = f"{uid}/"
    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            nrec = session.run(
                "MATCH (n:Entity) WHERE n.id STARTS WITH $p RETURN count(n) AS c",
                p=prefix,
            ).single()
            erec = session.run(
                """
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                WHERE a.id STARTS WITH $p AND b.id STARTS WITH $p
                RETURN count(r) AS c
                """,
                p=prefix,
            ).single()
            nodes = int(nrec["c"]) if nrec else 0
            edges = int(erec["c"]) if erec else 0
    finally:
        driver.close()
    return nodes, edges


def fetch_latest_scoring_record(user_id: str) -> dict[str, Any] | None:
    """Read the last NDJSON line from gs://.../<user_id>/scoring."""
    from google.cloud import storage

    from eval_agent.config import config, normalize_user_id, scoring_blob_path_for_user

    uid = normalize_user_id(user_id)
    bucket_name = (config.GCS_BUCKET_NAME or "").strip()
    if not bucket_name:
        return None
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(scoring_blob_path_for_user(uid))
    if not blob.exists():
        return None
    text = blob.download_as_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        logger.warning("Could not parse last line of scoring file as JSON.")
        return None


async def run_experiment(user_id: str, experiment_name: str | None) -> None:
    exp_name = experiment_name or "kg_pipeline_eval_fix"
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=f"{user_id}-{uuid.uuid4().hex[:8]}"):
        mlflow.log_param("user_id", kg_pipeline.normalize_user_id(user_id))

        # --- 1. Knowledge graph pipeline ---
        logger.info("Running KG pipeline for %s", user_id)
        pipeline_out = await kg_pipeline.run_pipeline(user_id)
        mlflow.log_dict(
            pipeline_out if pipeline_out else {"status": "no_return"},
            "kg_pipeline_result.json",
        )

        if pipeline_out:
            mlflow.log_param("pipeline_status", pipeline_out.get("status", "unknown"))
            if pipeline_out.get("status") == "completed":
                kr = pipeline_out.get("kg_result") or {}
                mlflow.log_metric("py_file_count", float(pipeline_out.get("py_file_count", 0)))
                mlflow.log_metric("summary_records", float(pipeline_out.get("summary_records", 0)))
                if isinstance(kr, dict):
                    mlflow.log_metric(
                        "kg_nodes_written_step",
                        float(kr.get("nodes_written", 0) or 0),
                    )
                    mlflow.log_metric(
                        "kg_edges_written_step",
                        float(kr.get("edges_written", 0) or 0),
                    )
                    mlflow.log_param("kg_transform_status", str(kr.get("status", "")))
        else:
            mlflow.log_param("pipeline_status", "none")

        # --- 2. Neo4j counts (per user id prefix) ---
        try:
            n_nodes, n_edges = neo4j_counts_for_user_prefix(user_id)
            mlflow.log_metric("neo4j_entity_nodes_for_user", float(n_nodes))
            mlflow.log_metric("neo4j_rel_edges_for_user", float(n_edges))
        except Exception as e:
            logger.exception("Neo4j count failed: %s", e)
            mlflow.log_param("neo4j_count_error", str(e)[:500])

        # --- 3. Eval agent ---
        logger.info("Running eval agent...")
        try:
            eval_text = await run_eval_agent_session(user_id)
            mlflow.log_text(eval_text or "(empty)", "eval_agent_output.txt")
        except Exception as e:
            logger.exception("Eval agent failed: %s", e)
            mlflow.log_text(str(e), "eval_agent_error.txt")

        # --- 4. Latest scoring row from GCS (written by save_scores_to_gcs) ---
        try:
            rec = fetch_latest_scoring_record(user_id)
            if rec:
                mlflow.log_dict(rec, "eval_scoring_record.json")
                for key in (
                    "criterion_1_score",
                    "criterion_2_score",
                    "overall_score",
                    "samples_evaluated",
                ):
                    if key in rec and rec[key] is not None:
                        try:
                            mlflow.log_metric(f"eval_{key}", float(rec[key]))
                        except (TypeError, ValueError):
                            pass
            else:
                mlflow.log_param("eval_scoring_gcs", "missing_or_empty")
        except Exception as e:
            logger.warning("Could not fetch scoring record: %s", e)
            mlflow.log_param("eval_scoring_fetch_error", str(e)[:500])


def main() -> None:
    p = argparse.ArgumentParser(
        description="MLflow run: KG pipeline (user_id) + eval agent + metrics."
    )
    p.add_argument(
        "--user-id",
        required=True,
        help="GCS / Neo4j prefix, e.g. u_123 (same as pipeline2 --user-id).",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name (default: kg_pipeline_eval).",
    )
    args = p.parse_args()
    asyncio.run(run_experiment(args.user_id, args.experiment_name))


if __name__ == "__main__":
    main()
