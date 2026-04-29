"""
Production incremental KG pipeline with AuraDB multi-tenant support.

Multi-Tenancy Strategy (AuraDB / single-database)
--------------------------------------------------
AuraDB Free/Professional does not support multiple physical databases on one
instance.  All users share a single ``neo4j`` database; isolation is enforced
entirely in software using two complementary mechanisms:

  1. **Label-Based Isolation** — every node gets an additional dynamic label
     ``:User_<uid>`` so Neo4j index scans jump directly to one tenant's
     sub-graph without inspecting other tenants' data.

  2. **Property-Based Isolation** — every node AND relationship carries a
     ``tenantId`` property equal to the normalized user ID.  Cypher MERGE keys
     include this property, so entity names that collide across tenants are
     stored as distinct nodes.

  Together these let GraphRAG queries scope to a single user with either pattern:

    -- Label-based (fast index scan):
    MATCH (n:Entity:User_u_123)-[r]->(m:Entity:User_u_123)
    WHERE n.name = "Config"
    RETURN n, r, m

    -- Property-based (portable, no schema knowledge required):
    MATCH (n:Entity {tenantId: 'u_123'})-[r {tenantId: 'u_123'}]->
          (m:Entity {tenantId: 'u_123'})
    RETURN n, r, m

Dual-Track KG Construction (from pipeline3)
--------------------------------------------
Each project is processed through two summarization + graph-extraction tracks:

  structural — summaries emphasise architecture, module boundaries, and
               cross-component dependency flow; uses larger overlapping batches
               so the LLM sees inter-file context.

  technical  — summaries emphasise concrete implementation: classes, functions,
               signatures, control flow, and algorithms; uses tight per-file
               batching for code-accurate entity extraction.

Each track caches its summaries under a separate GCS prefix and carries its
own ``kg_ingested`` flag, so incremental re-runs only process files that are
genuinely new or changed for that track.

Hyperparameters
---------------
Configurable via environment variables or CLI flags (CLI takes precedence):

  PIPELINE_STRUCTURAL_KG_BATCH       / --structural-batch-size       (default 4)
  PIPELINE_STRUCTURAL_KG_OVERLAP     / --structural-batch-overlap     (default 2)
  PIPELINE_STRUCTURAL_MAX_DOC_CHARS  / --structural-max-doc-chars     (default 1600)
  PIPELINE_TECHNICAL_KG_BATCH        / --technical-batch-size         (default 1)
  PIPELINE_TECHNICAL_KG_OVERLAP      / --technical-batch-overlap      (default 0)
  PIPELINE_TECHNICAL_MAX_DOC_CHARS   / --technical-max-doc-chars      (default 2800)

Run:
  python prod_pipeline.py --user-id u_123
  python prod_pipeline.py --user-id u_123 --structural-batch-size 6 \\
      --technical-max-doc-chars 3200 --neo4j-database neo4j
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any

from config import Config
from google.cloud import storage
from pipeline2 import (
    code_prefix_for_user,
    load_code_files_for_names,
    load_summary_record,
    normalize_user_id,
    run_parallel_summaries,
    summary_json_exists,
    upload_summary_record,
)
from tools2 import build_kg_and_push_to_neo4j

logger = logging.getLogger("Orchestrator.prod_pipeline")

# ── Summary folder names (GCS prefixes) ────────────────────────────────────
SUMMARIES_STRUCTURAL = "summaries_structural"
SUMMARIES_TECHNICAL = "summaries_technical"

# ── Ingestion tracking keys stored inside each summary JSON ────────────────
INGESTION_FLAG = "kg_ingested"
INGESTION_PIPELINE = "kg_ingested_by"
INGESTION_TS = "kg_ingested_at"
PIPELINE_NAME = "prod_pipeline_v2"


# ── Env-driven hyperparameter defaults ─────────────────────────────────────
def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


DEFAULT_STRUCTURAL_KG_BATCH = _env_int("PIPELINE_STRUCTURAL_KG_BATCH", 4)
DEFAULT_STRUCTURAL_KG_OVERLAP = _env_int("PIPELINE_STRUCTURAL_KG_OVERLAP", 2)
DEFAULT_STRUCTURAL_MAX_DOC_CHARS = _env_int("PIPELINE_STRUCTURAL_MAX_DOC_CHARS", 1600)
DEFAULT_TECHNICAL_KG_BATCH = _env_int("PIPELINE_TECHNICAL_KG_BATCH", 1)
DEFAULT_TECHNICAL_KG_OVERLAP = _env_int("PIPELINE_TECHNICAL_KG_OVERLAP", 0)
DEFAULT_TECHNICAL_MAX_DOC_CHARS = _env_int("PIPELINE_TECHNICAL_MAX_DOC_CHARS", 2800)


# ── Summarizer prompts ─────────────────────────────────────────────────────
STRUCTURAL_SUMMARIZER_INSTRUCTION = (
    "You are a senior software architect. From Python source, produce a concise summary "
    "that emphasises how the codebase is structured: modules, layers, responsibilities, "
    "and dependencies between parts. Capture cross-cutting relationships and design intent."
)
STRUCTURAL_FOCUS_SUFFIX = (
    "\n\nFocus especially on: high-level structure, module boundaries, dependency flow "
    "between components, and how the organisation reflects the author's knowledge of "
    "designing and structuring a codebase."
)

TECHNICAL_SUMMARIZER_INSTRUCTION = (
    "You are a technical code analyst. From the Python source below, produce a detailed "
    "summary grounded in the actual implementation: concrete classes, functions, APIs, "
    "control flow, data structures, error handling, and algorithms. Quote or paraphrase "
    "behaviours evidenced in the code."
)
TECHNICAL_USER_PREFIX = (
    "File: {file_name}\n\n"
    "The following is the full Python source. Base your analysis strictly on this code.\n\n"
    "{file_content}"
)
TECHNICAL_FOCUS_SUFFIX = (
    "\n\nFocus especially on: implementation-level technical understanding — signatures, "
    "branching, side effects, and dependencies visible in the source itself."
)


# ── GCS helpers ────────────────────────────────────────────────────────────
def _summary_blob_path(code_prefix: str, code_blob_name: str, track_folder: str) -> str:
    if not code_blob_name.startswith(code_prefix):
        raise ValueError(f"code blob {code_blob_name!r} must start with {code_prefix!r}")
    rel = code_blob_name[len(code_prefix):]
    return f"{code_prefix}{track_folder}/{rel}.json"


def _list_python_blob_names(bucket: storage.Bucket, code_prefix: str) -> list[str]:
    """List .py blobs under code_prefix, skipping both summary cache folders."""
    out: list[str] = []
    guarded = {
        f"{code_prefix}{SUMMARIES_STRUCTURAL}/",
        f"{code_prefix}{SUMMARIES_TECHNICAL}/",
    }
    for blob in bucket.list_blobs(prefix=code_prefix):
        name = blob.name
        if any(name.startswith(g) for g in guarded):
            continue
        if name.endswith(".py"):
            out.append(name)
    return sorted(out)


def _project_key(code_prefix: str, code_blob_name: str) -> str:
    rel = code_blob_name[len(code_prefix):]
    if "/" not in rel:
        return "__root__"
    return rel.split("/", 1)[0] or "__root__"


def _group_files_by_project(code_prefix: str, py_blob_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in py_blob_names:
        key = _project_key(code_prefix, name)
        grouped.setdefault(key, []).append(name)
    return grouped


# ── Summary record helpers ─────────────────────────────────────────────────
def _summary_record_is_usable(rec: dict[str, Any] | None) -> bool:
    return bool(rec and rec.get("status") == "completed" and rec.get("summary"))


def _summary_needs_generation(
    bucket: storage.Bucket,
    code_prefix: str,
    code_blob_name: str,
    track_folder: str,
) -> bool:
    sp = _summary_blob_path(code_prefix, code_blob_name, track_folder)
    if not summary_json_exists(bucket, sp):
        return True
    rec = load_summary_record(bucket, sp)
    return not _summary_record_is_usable(rec)


def _summary_is_ingested(rec: dict[str, Any] | None) -> bool:
    if not _summary_record_is_usable(rec):
        return False
    return bool(rec.get(INGESTION_FLAG))


def _mark_ingested(
    bucket: storage.Bucket,
    summary_path: str,
    rec: dict[str, Any],
) -> None:
    out = dict(rec)
    out[INGESTION_FLAG] = True
    out[INGESTION_PIPELINE] = PIPELINE_NAME
    out[INGESTION_TS] = datetime.now(timezone.utc).isoformat()
    upload_summary_record(bucket, summary_path, out)


def _build_cached_result(name: str, rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "file": rec.get("file", name),
        "summary": str(rec["summary"]),
        "status": "completed",
    }


# ── Per-track summary generation + ingestion check ────────────────────────
async def _ensure_summaries_for_track(
    *,
    bucket: storage.Bucket,
    code_prefix: str,
    files: list[str],
    track_folder: str,
    track_key: str,
    model_name: str,
    max_parallel: int,
    adk_user: str,
    summarizer_instruction: str,
    summary_focus_suffix: str,
    user_message_prefix: str | None,
) -> tuple[list[str], list[str]]:
    """
    Generate missing summaries for *files* under *track_folder*, then return
    ``(generated_now, ready_for_kg)`` where ``ready_for_kg`` contains every file
    that has a usable summary but has NOT yet been ingested for this track.
    """
    needs_generation = [
        name for name in files
        if _summary_needs_generation(bucket, code_prefix, name, track_folder)
    ]
    if needs_generation:
        to_generate = load_code_files_for_names(bucket, needs_generation)
        if len(to_generate) != len(needs_generation):
            missing = sorted(set(needs_generation) - {x["file_name"] for x in to_generate})
            raise RuntimeError(f"[{track_key}] Could not load code blobs: {missing}")

        fresh_list = await run_parallel_summaries(
            to_generate,
            model_name,
            max_parallel,
            adk_user,
            summarizer_instruction=summarizer_instruction,
            user_message_prefix=user_message_prefix,
            summary_focus_suffix=summary_focus_suffix,
        )
        fresh_by_file = {r["file"]: r for r in fresh_list}
        for name in needs_generation:
            res = fresh_by_file.get(name) or {"file": name, "summary": None, "status": "failed"}
            sp = _summary_blob_path(code_prefix, name, track_folder)
            upload_summary_record(
                bucket,
                sp,
                {
                    "file": name,
                    "summary": res.get("summary"),
                    "status": res.get("status"),
                    "error": res.get("error"),
                    "track": track_key,
                },
            )
            logger.info("[%s] Wrote summary to gs://%s/%s", track_key, bucket.name, sp)

    ready_for_kg: list[str] = []
    for name in files:
        sp = _summary_blob_path(code_prefix, name, track_folder)
        rec = load_summary_record(bucket, sp)
        if not _summary_record_is_usable(rec):
            continue
        if _summary_is_ingested(rec):
            continue
        ready_for_kg.append(name)

    return needs_generation, ready_for_kg


def _kg_results_for_files(
    bucket: storage.Bucket,
    code_prefix: str,
    files: list[str],
    track_folder: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in files:
        sp = _summary_blob_path(code_prefix, name, track_folder)
        rec = load_summary_record(bucket, sp)
        if not _summary_record_is_usable(rec):
            logger.warning("[%s] Skipping unusable summary %s", track_folder, sp)
            continue
        out.append(_build_cached_result(name, rec))
    return out


# ── Single track: summarise + KG build for one project ────────────────────
async def _run_track_for_project(
    *,
    bucket: storage.Bucket,
    code_prefix: str,
    project: str,
    files: list[str],
    track_key: str,
    track_folder: str,
    model_name: str,
    max_parallel: int,
    uid: str,
    summarizer_instruction: str,
    summary_focus_suffix: str,
    user_message_prefix: str | None,
    kg_batch_size: int,
    kg_batch_overlap: int,
    kg_max_doc_chars: int,
    extra_entity_props: dict[str, Any],
    neo4j_database: str | None,
    tenant_id: str,
) -> dict[str, Any]:
    """
    Run summarisation (if needed) and KG ingestion for one track of one project.
    Already-ingested files are silently skipped; only new ones are sent to Neo4j.
    """
    adk_user = f"kg-prod-{track_key}-{uid}-{project}"

    generated_now, ready_for_kg = await _ensure_summaries_for_track(
        bucket=bucket,
        code_prefix=code_prefix,
        files=files,
        track_folder=track_folder,
        track_key=track_key,
        model_name=model_name,
        max_parallel=max_parallel,
        adk_user=adk_user,
        summarizer_instruction=summarizer_instruction,
        summary_focus_suffix=summary_focus_suffix,
        user_message_prefix=user_message_prefix,
    )

    if not ready_for_kg:
        return {
            "track": track_key,
            "status": "completed",
            "summaries_generated_now": len(generated_now),
            "files_sent_to_kg": 0,
            "kg_result": {"status": "skipped", "message": "All summaries already ingested."},
        }

    kg_results = _kg_results_for_files(bucket, code_prefix, ready_for_kg, track_folder)
    if not kg_results:
        return {
            "track": track_key,
            "status": "error",
            "reason": "no_usable_summaries_for_kg",
            "summaries_generated_now": len(generated_now),
            "files_sent_to_kg": 0,
        }

    logger.info(
        "[project=%s][%s] Building KG for %d summary record(s), tenant=%s",
        project, track_key, len(kg_results), tenant_id,
    )
    kg_out = build_kg_and_push_to_neo4j(
        kg_results,
        kg_batch_size=kg_batch_size,
        kg_batch_overlap=kg_batch_overlap,
        kg_max_doc_chars=kg_max_doc_chars,
        extra_entity_props={**extra_entity_props, "kg_project": project},
        neo4j_database=neo4j_database,
        tenant_id=tenant_id,
    )

    if kg_out.get("status") == "success":
        for name in ready_for_kg:
            sp = _summary_blob_path(code_prefix, name, track_folder)
            rec = load_summary_record(bucket, sp) or {
                "file": name, "status": "completed", "summary": "",
            }
            _mark_ingested(bucket, sp, rec)

    return {
        "track": track_key,
        "status": "completed" if kg_out.get("status") == "success" else "error",
        "summaries_generated_now": len(generated_now),
        "files_sent_to_kg": len(kg_results),
        "kg_result": kg_out,
    }


# ── Main pipeline entry point ──────────────────────────────────────────────
async def run_prod_pipeline(
    user_id: str,
    *,
    neo4j_database: str | None = None,
    structural_batch_size: int | None = None,
    structural_batch_overlap: int | None = None,
    structural_max_doc_chars: int | None = None,
    technical_batch_size: int | None = None,
    technical_batch_overlap: int | None = None,
    technical_max_doc_chars: int | None = None,
) -> dict[str, Any]:
    """
    Incremental dual-track KG pipeline scoped to a single AuraDB instance via
    label + property tenancy (``tenant_id`` = normalised ``user_id``).

    Behaviour:
    - Discover Python files under ``gs://<bucket>/<user_id>/``.
    - Group files into projects (first sub-folder under ``<user_id>/``).
    - For each project, run **structural** then **technical** tracks.
    - Files already ingested for a given track are skipped on every re-run.
    - New files are summarised and merged into the tenant-scoped graph.
    - Hyperparameter defaults come from env vars; CLI args override them.
    """
    uid = normalize_user_id(user_id)
    bucket_name = Config.GCS_BUCKET_NAME
    code_prefix = code_prefix_for_user(uid)
    model_name = Config.GEMINI_MODEL
    max_parallel = Config.SUMMARY_MAX_PARALLEL

    s_batch = structural_batch_size if structural_batch_size is not None else DEFAULT_STRUCTURAL_KG_BATCH
    s_overlap = structural_batch_overlap if structural_batch_overlap is not None else DEFAULT_STRUCTURAL_KG_OVERLAP
    s_chars = structural_max_doc_chars if structural_max_doc_chars is not None else DEFAULT_STRUCTURAL_MAX_DOC_CHARS
    t_batch = technical_batch_size if technical_batch_size is not None else DEFAULT_TECHNICAL_KG_BATCH
    t_overlap = technical_batch_overlap if technical_batch_overlap is not None else DEFAULT_TECHNICAL_KG_OVERLAP
    t_chars = technical_max_doc_chars if technical_max_doc_chars is not None else DEFAULT_TECHNICAL_MAX_DOC_CHARS

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    py_blob_names = _list_python_blob_names(bucket, code_prefix)
    if not py_blob_names:
        return {"status": "skipped", "reason": "no_py_files", "user_id": uid}

    grouped = _group_files_by_project(code_prefix, py_blob_names)
    project_reports: dict[str, Any] = {}
    total_nodes = 0
    total_edges = 0

    for project, files in sorted(grouped.items()):
        logger.info(
            "[project=%s] %d file(s) | tenant=%s | tracks: structural + technical",
            project, len(files), uid,
        )

        # Structural first — broader batches give the LLM cross-file context
        # before technical extraction drills into per-file implementation detail.
        structural_report = await _run_track_for_project(
            bucket=bucket,
            code_prefix=code_prefix,
            project=project,
            files=files,
            track_key="structural",
            track_folder=SUMMARIES_STRUCTURAL,
            model_name=model_name,
            max_parallel=max_parallel,
            uid=uid,
            summarizer_instruction=STRUCTURAL_SUMMARIZER_INSTRUCTION,
            summary_focus_suffix=STRUCTURAL_FOCUS_SUFFIX,
            user_message_prefix=None,
            kg_batch_size=s_batch,
            kg_batch_overlap=s_overlap,
            kg_max_doc_chars=s_chars,
            extra_entity_props={"kg_track_structural": True},
            neo4j_database=neo4j_database,
            tenant_id=uid,
        )
        technical_report = await _run_track_for_project(
            bucket=bucket,
            code_prefix=code_prefix,
            project=project,
            files=files,
            track_key="technical",
            track_folder=SUMMARIES_TECHNICAL,
            model_name=model_name,
            max_parallel=max_parallel,
            uid=uid,
            summarizer_instruction=TECHNICAL_SUMMARIZER_INSTRUCTION,
            summary_focus_suffix=TECHNICAL_FOCUS_SUFFIX,
            user_message_prefix=TECHNICAL_USER_PREFIX,
            kg_batch_size=t_batch,
            kg_batch_overlap=t_overlap,
            kg_max_doc_chars=t_chars,
            extra_entity_props={"kg_track_technical": True},
            neo4j_database=neo4j_database,
            tenant_id=uid,
        )

        for report in (structural_report, technical_report):
            kg_r = report.get("kg_result") or {}
            total_nodes += int(kg_r.get("nodes_written") or 0)
            total_edges += int(kg_r.get("edges_written") or 0)

        project_reports[project] = {
            "status": "completed",
            "file_count": len(files),
            "structural": structural_report,
            "technical": technical_report,
        }

    out: dict[str, Any] = {
        "status": "completed",
        "user_id": uid,
        "tenant_id": uid,
        "py_file_count": len(py_blob_names),
        "project_count": len(grouped),
        "nodes_written_total": total_nodes,
        "edges_written_total": total_edges,
        "hyperparameters": {
            "structural": {
                "kg_batch_size": s_batch,
                "kg_batch_overlap": s_overlap,
                "kg_max_doc_chars": s_chars,
            },
            "technical": {
                "kg_batch_size": t_batch,
                "kg_batch_overlap": t_overlap,
                "kg_max_doc_chars": t_chars,
            },
        },
        "projects": project_reports,
    }
    if neo4j_database:
        out["neo4j_database"] = neo4j_database
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    p = argparse.ArgumentParser(
        description=(
            "Incremental dual-track KG pipeline with AuraDB multi-tenancy. "
            "Runs structural (architecture) and technical (implementation) tracks per project. "
            "New files are merged into the tenant-scoped graph; already-ingested files are skipped."
        )
    )
    p.add_argument(
        "--user-id",
        required=True,
        help="GCS prefix / tenant ID, e.g. u_123",
    )
    p.add_argument(
        "--neo4j-database",
        default=None,
        dest="neo4j_database",
        metavar="NAME",
        help="Neo4j logical database name. Omit to use server default (neo4j on AuraDB).",
    )

    # ── Structural track hyperparameters ──────────────────────────────────
    p.add_argument(
        "--structural-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Number of file summaries per KG extraction batch for the structural track. "
            f"Larger batches let the LLM see more cross-file context. "
            f"Default: {DEFAULT_STRUCTURAL_KG_BATCH} (env: PIPELINE_STRUCTURAL_KG_BATCH)."
        ),
    )
    p.add_argument(
        "--structural-batch-overlap",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Number of documents shared between adjacent structural batches. "
            f"Overlap helps preserve context at batch boundaries. "
            f"Default: {DEFAULT_STRUCTURAL_KG_OVERLAP} (env: PIPELINE_STRUCTURAL_KG_OVERLAP)."
        ),
    )
    p.add_argument(
        "--structural-max-doc-chars",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Hard character cap applied to each structural summary before it is fed "
            f"to the graph transformer. Prevents token-limit errors on long summaries. "
            f"Default: {DEFAULT_STRUCTURAL_MAX_DOC_CHARS} (env: PIPELINE_STRUCTURAL_MAX_DOC_CHARS)."
        ),
    )

    # ── Technical track hyperparameters ───────────────────────────────────
    p.add_argument(
        "--technical-batch-size",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Number of file summaries per KG extraction batch for the technical track. "
            f"Defaults to 1 so each file is extracted independently for code accuracy. "
            f"Default: {DEFAULT_TECHNICAL_KG_BATCH} (env: PIPELINE_TECHNICAL_KG_BATCH)."
        ),
    )
    p.add_argument(
        "--technical-batch-overlap",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Batch overlap for the technical track. Usually 0 since batches are per-file. "
            f"Default: {DEFAULT_TECHNICAL_KG_OVERLAP} (env: PIPELINE_TECHNICAL_KG_OVERLAP)."
        ),
    )
    p.add_argument(
        "--technical-max-doc-chars",
        type=int,
        default=None,
        metavar="N",
        help=(
            f"Hard character cap for technical summaries. Set higher than structural to "
            f"preserve implementation detail. "
            f"Default: {DEFAULT_TECHNICAL_MAX_DOC_CHARS} (env: PIPELINE_TECHNICAL_MAX_DOC_CHARS)."
        ),
    )

    args = p.parse_args()
    db = (args.neo4j_database or "").strip() or None
    out = asyncio.run(
        run_prod_pipeline(
            args.user_id,
            neo4j_database=db,
            structural_batch_size=args.structural_batch_size,
            structural_batch_overlap=args.structural_batch_overlap,
            structural_max_doc_chars=args.structural_max_doc_chars,
            technical_batch_size=args.technical_batch_size,
            technical_batch_overlap=args.technical_batch_overlap,
            technical_max_doc_chars=args.technical_max_doc_chars,
        )
    )
    logger.info("prod_pipeline finished: %s", out)


if __name__ == "__main__":
    main()
