"""
MLflow experiment: build a knowledge graph (KG_agent pipeline3) into a **fresh Neo4j database**
per run (random 6-char name, e.g. ``axd56f``), then run eval_agent (ADK) against that database. Logs summarizer + KG graph model
metadata, pipeline output, Neo4j counts (plus per relationship-type and entity-kind
counts), eval transcript, and scoring metrics from GCS.

Run from Ops_Project (repo root for imports):

  cd Ops_Project
  python experiment.py --user-id u_123

Requires: same .env as KG_agent / eval_agent (Neo4j multi-database or CREATE DATABASE permission,
GCS, Vertex, MLflow tracking URI optional).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import secrets
import string
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
import pipeline3 as kg_pipeline  # noqa: E402
from pipeline2 import normalize_user_id  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("experiment")


def _sanitize_neo4j_database_name(raw: str) -> str:
    """Neo4j logical DB names: letters, digits, underscore; must start with a letter."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "").strip())
    if not s:
        s = "graph"
    if not s[0].isalpha():
        s = "g_" + s
    # Keep names reasonably short for Neo4j admin UIs
    if len(s) > 48:
        s = s[:48]
    return s


def neo4j_experiment_database_name() -> str:
    """
    Random 6-character Neo4j logical database name (first char is a letter), e.g. ``axd56f``.
    Neo4j requires names to start with a letter; we use lowercase letters + digits.
    """
    letters = string.ascii_lowercase
    alnum = letters + string.digits
    first = secrets.choice(letters)
    rest = "".join(secrets.choice(alnum) for _ in range(5))
    return _sanitize_neo4j_database_name(first + rest)


def ensure_neo4j_database_exists(database: str) -> None:
    """
    Best-effort ``CREATE DATABASE ... IF NOT EXISTS`` against the ``system`` database.
    Skips if this fails (e.g. Community / permissions); the run may still succeed if the DB exists.
    """
    from config import Config
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )
    try:
        with driver.session(database="system") as session:
            session.run(f"CREATE DATABASE `{database}` IF NOT EXISTS")
        logger.info("Ensured Neo4j database exists: %s", database)
    except Exception as e:
        logger.warning(
            "Could not CREATE DATABASE `%s` via system session (%s). "
            "If the database does not already exist, pipeline writes may fail.",
            database,
            e,
        )
    finally:
        driver.close()


def log_kg_model_params() -> None:
    """Record which models/backends are configured for summaries and graph extraction."""
    from config import Config as KGConfig

    mlflow.log_param("summary_llm_model", KGConfig.GEMINI_MODEL)
    mlflow.log_param("kg_graph_backend", KGConfig.KG_GRAPH_BACKEND)
    mlflow.log_param("kg_vertex_graph_model", KGConfig.VERTEX_GEMINI_MODEL)
    mlflow.log_param("kg_hf_graph_model", KGConfig.HF_GRAPH_MODEL)


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


async def run_eval_agent_session(user_id: str, neo4j_database: str) -> str:
    """ADK eval session with one retry on silent final response."""
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
    adk_user = f"experiment-{user_id}"
    prompt = (
        f"My user_id is `{user_id}` (for GCS scoring paths). "
        f"The Neo4j **database name** for this experiment graph is `{neo4j_database}` — "
        "you **must** pass this exact string as `neo4j_database` to **every** tool that queries Neo4j "
        "(`list_entity_id_samples`, `fetch_neighborhood_from_neo4j`). "
        "Do not use user_id as the Neo4j database unless it happens to match.\n\n"
        "Perform the complete knowledge-graph evaluation: sample several neighborhoods, "
        "ground with retrieve_from_rag_corpus, assign per-sample scores, aggregate averages, "
        "output the ## Final scores block, then call save_scores_to_gcs with user_id, "
        "per_sample_scores_json, and all required fields."
    )
    msg = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        session_id = f"exp-{uuid.uuid4().hex}"
        text = await _final_text_from_events(
            runner.run_async(
                user_id=adk_user,
                session_id=session_id,
                new_message=msg,
            )
        )
        if text and text.strip():
            return text
        if attempt < max_attempts:
            logger.warning(
                "Eval agent returned silent response; retrying (attempt %d/%d)",
                attempt + 1,
                max_attempts,
            )
    return ""


def _safe_mlflow_metric_suffix(raw: str, *, max_len: int = 96) -> str:
    """Metric keys must be stable; strip characters MLflow / UIs handle poorly."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "unknown").strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "unknown")[:max_len]


def neo4j_graph_metadata_for_user_prefix(
    user_id: str, neo4j_database: str
) -> dict[str, Any]:
    """
    Aggregate :Entity / :REL stats for ids under ``<user_id>/`` in the given logical database.

    Returns entity and edge totals, counts per ``r.type`` on :REL, and per ``n.kind`` on :Entity.
    """
    from config import Config
    from neo4j import GraphDatabase

    uid = normalize_user_id(user_id)
    prefix = f"{uid}/"
    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )
    out: dict[str, Any] = {
        "entity_nodes": 0,
        "rel_edges": 0,
        "relationship_type_counts": {},
        "entity_kind_counts": {},
    }
    try:
        with driver.session(database=neo4j_database) as session:
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
            out["entity_nodes"] = int(nrec["c"]) if nrec else 0
            out["rel_edges"] = int(erec["c"]) if erec else 0

            for row in session.run(
                """
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                WHERE a.id STARTS WITH $p AND b.id STARTS WITH $p
                RETURN coalesce(r.type, '(null)') AS rel_type, count(*) AS c
                """,
                p=prefix,
            ):
                key = str(row["rel_type"])
                out["relationship_type_counts"][key] = int(row["c"])

            for row in session.run(
                """
                MATCH (n:Entity)
                WHERE n.id STARTS WITH $p
                RETURN coalesce(n.kind, '(null)') AS kind, count(*) AS c
                """,
                p=prefix,
            ):
                key = str(row["kind"])
                out["entity_kind_counts"][key] = int(row["c"])
    finally:
        driver.close()
    return out


def neo4j_counts_for_user_prefix(user_id: str, neo4j_database: str) -> tuple[int, int]:
    """Count :Entity and :REL where entity ids start with ``<user_id>/`` in the given database."""
    meta = neo4j_graph_metadata_for_user_prefix(user_id, neo4j_database)
    return int(meta["entity_nodes"]), int(meta["rel_edges"])


def log_neo4j_schema_to_mlflow(meta: dict[str, Any]) -> None:
    """Log KG shape (rel types, entity kinds) as JSON artifact and per-key metrics."""
    mlflow.log_dict(
        {
            "relationship_type_counts": meta.get("relationship_type_counts") or {},
            "entity_kind_counts": meta.get("entity_kind_counts") or {},
        },
        "neo4j_graph_schema_counts.json",
    )
    rel_counts: dict[str, Any] = meta.get("relationship_type_counts") or {}
    for rel_type, count in sorted(rel_counts.items(), key=lambda x: (-x[1], x[0])):
        suffix = _safe_mlflow_metric_suffix(rel_type)
        mlflow.log_metric(f"neo4j_rel_type_{suffix}", float(count))
    kind_counts: dict[str, Any] = meta.get("entity_kind_counts") or {}
    for kind, count in sorted(kind_counts.items(), key=lambda x: (-x[1], x[0])):
        suffix = _safe_mlflow_metric_suffix(kind)
        mlflow.log_metric(f"neo4j_entity_kind_{suffix}", float(count))


def fetch_latest_scoring_record(user_id: str) -> dict[str, Any] | None:
    """Read the last NDJSON line from gs://.../<user_id>/scoring."""
    from google.cloud import storage

    from eval_agent.config import (
        config,
        normalize_user_id as eval_normalize_user_id,
        scoring_blob_path_for_user,
    )

    uid = eval_normalize_user_id(user_id)
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


def _log_pipeline3_metrics(pipeline_out: dict[str, Any]) -> None:
    """Extract metrics from pipeline3 dual-track result."""
    mlflow.log_param("pipeline_status", pipeline_out.get("status", "unknown"))
    if pipeline_out.get("neo4j_database"):
        mlflow.log_param("pipeline_neo4j_database", str(pipeline_out["neo4j_database"]))

    if pipeline_out.get("status") != "completed":
        return

    mlflow.log_metric("py_file_count", float(pipeline_out.get("py_file_count", 0)))

    for track_key in ("structural", "technical"):
        tr = pipeline_out.get(track_key)
        if not isinstance(tr, dict):
            continue
        mlflow.log_metric(
            f"summary_records_{track_key}",
            float(tr.get("summary_records", 0) or 0),
        )
        kr = tr.get("kg_result")
        if isinstance(kr, dict):
            mlflow.log_metric(
                f"kg_{track_key}_nodes_written",
                float(kr.get("nodes_written", 0) or 0),
            )
            mlflow.log_metric(
                f"kg_{track_key}_edges_written",
                float(kr.get("edges_written", 0) or 0),
            )
            mlflow.log_param(
                f"kg_{track_key}_transform_status",
                str(kr.get("status", "")),
            )
            ktu = kr.get("token_usage")
            if isinstance(ktu, dict):
                mlflow.log_param(
                    f"kg_{track_key}_model",
                    str(ktu.get("model", "")),
                )
                mlflow.log_param(
                    f"kg_{track_key}_token_method",
                    str(ktu.get("method", "")),
                )
                mlflow.log_metric(
                    f"kg_{track_key}_input_tokens_est",
                    float(ktu.get("input_tokens_est", 0) or 0),
                )
                mlflow.log_metric(
                    f"kg_{track_key}_output_tokens_est",
                    float(ktu.get("output_tokens_est", 0) or 0),
                )
                mlflow.log_metric(
                    f"kg_{track_key}_total_tokens_est",
                    float(ktu.get("total_tokens_est", 0) or 0),
                )

        stu = tr.get("summary_token_usage")
        if isinstance(stu, dict):
            mlflow.log_param(
                f"summary_{track_key}_model",
                str(stu.get("model", "")),
            )
            mlflow.log_param(
                f"summary_{track_key}_token_method",
                str(stu.get("method", "")),
            )
            mlflow.log_metric(
                f"summary_{track_key}_prompt_tokens_est",
                float(stu.get("prompt_tokens_est", 0) or 0),
            )
            mlflow.log_metric(
                f"summary_{track_key}_completion_tokens_est",
                float(stu.get("completion_tokens_est", 0) or 0),
            )
            mlflow.log_metric(
                f"summary_{track_key}_total_tokens_est",
                float(stu.get("total_tokens_est", 0) or 0),
            )

    # Run-level totals for quick cost estimation dashboards.
    total_summary_tokens_est = 0.0
    total_kg_tokens_est = 0.0
    for track_key in ("structural", "technical"):
        tr = pipeline_out.get(track_key)
        if not isinstance(tr, dict):
            continue
        stu = tr.get("summary_token_usage")
        if isinstance(stu, dict):
            total_summary_tokens_est += float(stu.get("total_tokens_est", 0) or 0)
        kr = tr.get("kg_result")
        if isinstance(kr, dict):
            ktu = kr.get("token_usage")
            if isinstance(ktu, dict):
                total_kg_tokens_est += float(ktu.get("total_tokens_est", 0) or 0)
    mlflow.log_metric("summary_total_tokens_est", total_summary_tokens_est)
    mlflow.log_metric("kg_total_tokens_est", total_kg_tokens_est)
    mlflow.log_metric(
        "experiment_total_tokens_est",
        total_summary_tokens_est + total_kg_tokens_est,
    )


async def run_experiment(user_id: str, experiment_name: str | None) -> None:
    exp_name = experiment_name or "kg_pipeline_eval_fix"
    mlflow.set_experiment(exp_name)

    uid_norm = normalize_user_id(user_id)
    neo4j_db = neo4j_experiment_database_name()

    with mlflow.start_run(run_name=f"{uid_norm}-{uuid.uuid4().hex[:8]}"):
        mlflow.log_param("user_id", uid_norm)
        mlflow.log_param("neo4j_database", neo4j_db)
        log_kg_model_params()

        ensure_neo4j_database_exists(neo4j_db)

        # --- 1. Knowledge graph pipeline (pipeline3: structural + technical tracks) ---
        logger.info("Running KG pipeline3 for %s → Neo4j database %s", user_id, neo4j_db)
        pipeline_out = await kg_pipeline.run_pipeline3(
            user_id,
            neo4j_database=neo4j_db,
        )
        mlflow.log_dict(
            pipeline_out if pipeline_out else {"status": "no_return"},
            "kg_pipeline_result.json",
        )

        if pipeline_out:
            _log_pipeline3_metrics(pipeline_out)
        else:
            mlflow.log_param("pipeline_status", "none")

        # --- 2. Neo4j counts + rel/kind distributions (per user id prefix, experiment DB) ---
        try:
            graph_meta = neo4j_graph_metadata_for_user_prefix(user_id, neo4j_db)
            mlflow.log_metric(
                "neo4j_entity_nodes_for_user", float(graph_meta["entity_nodes"])
            )
            mlflow.log_metric(
                "neo4j_rel_edges_for_user", float(graph_meta["rel_edges"])
            )
            log_neo4j_schema_to_mlflow(graph_meta)
        except Exception as e:
            logger.exception("Neo4j count failed: %s", e)
            mlflow.log_param("neo4j_count_error", str(e)[:500])

        # --- 3. Eval agent ---
        logger.info("Running eval agent (neo4j_database=%s)...", neo4j_db)
        try:
            eval_text = await run_eval_agent_session(user_id, neo4j_db)
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
        description="MLflow run: KG pipeline3 (fresh Neo4j DB per run) + eval agent + metrics."
    )
    p.add_argument(
        "--user-id",
        required=True,
        help="GCS prefix, e.g. u_123 (same as pipeline --user-id).",
    )
    p.add_argument(
        "--experiment-name",
        default=None,
        help="MLflow experiment name (default: kg_pipeline_eval_fix).",
    )
    args = p.parse_args()
    asyncio.run(run_experiment(args.user_id, args.experiment_name))


if __name__ == "__main__":
    main()
