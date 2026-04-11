"""
Dual-track knowledge-graph pipeline: merge **structural** and **technical** graphs in Neo4j.

1. **Structural / overview track** — Summaries emphasize architecture, layering, and
   cross-module dependencies; larger batch overlap so batched graph extraction sees more
   inter-file context. Entities tagged ``kg_track: structural``.

2. **Technical track** — Summaries emphasize implementation detail grounded in the ``.py``
   source; per-file (or tight) batching for code-accurate entities. Entities tagged
   ``kg_track: technical``.

Caches per-track summaries under:
  ``gs://<bucket>/<user_id>/summaries_structural/<path>.py.json``
  ``gs://<bucket>/<user_id>/summaries_technical/<path>.py.json``

Run (from ``KG_agent``):

  python pipeline3.py --user-id u_123
  python pipeline3.py --user-id u_123 --neo4j-database mygraph
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Any

from config import Config
from google.cloud import storage
from tools2 import build_kg_and_push_to_neo4j

from pipeline2 import (
    code_prefix_for_user,
    list_python_blob_names,
    load_code_files_for_names,
    merge_results_for_kg,
    normalize_user_id,
    run_parallel_summaries,
    upload_summary_record,
    load_summary_record,
    summary_json_exists,
)

logger = logging.getLogger("Orchestrator.pipeline3")

# --- Summary storage (separate from pipeline2's summaries/) ---
SUMMARIES_STRUCTURAL = "summaries_structural"
SUMMARIES_TECHNICAL = "summaries_technical"

# --- Tunable via environment (optional) ---
def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


STRUCTURAL_KG_BATCH = _env_int("PIPELINE3_STRUCTURAL_KG_BATCH", 4)
STRUCTURAL_KG_OVERLAP = _env_int("PIPELINE3_STRUCTURAL_KG_OVERLAP", 2)
STRUCTURAL_MAX_DOC_CHARS = _env_int("PIPELINE3_STRUCTURAL_MAX_DOC_CHARS", 1600)

TECHNICAL_KG_BATCH = _env_int("PIPELINE3_TECHNICAL_KG_BATCH", 1)
TECHNICAL_KG_OVERLAP = _env_int("PIPELINE3_TECHNICAL_KG_OVERLAP", 0)
TECHNICAL_MAX_DOC_CHARS = _env_int("PIPELINE3_TECHNICAL_MAX_DOC_CHARS", 2800)


STRUCTURAL_SUMMARIZER_INSTRUCTION = (
    "You are a senior software architect. From Python source, produce a concise summary "
    "that emphasizes how the codebase is structured: modules, layers, responsibilities, "
    "and dependencies between parts. Capture cross-cutting relationships and design intent."
)

STRUCTURAL_FOCUS_SUFFIX = (
    "\n\nFocus especially on: high-level structure, module boundaries, dependency flow "
    "between components, and how the organization reflects the author's knowledge of "
    "designing and structuring a codebase."
)

TECHNICAL_SUMMARIZER_INSTRUCTION = (
    "You are a technical code analyst. From the Python source below, produce a detailed "
    "summary grounded in the actual implementation: concrete classes, functions, APIs, "
    "control flow, data structures, error handling, and algorithms. Quote or paraphrase "
    "behaviors evidenced in the code."
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


def summary_blob_path_track(code_prefix: str, code_blob_name: str, track_folder: str) -> str:
    if not code_blob_name.startswith(code_prefix):
        raise ValueError(f"code blob {code_blob_name!r} must start with {code_prefix!r}")
    rel = code_blob_name[len(code_prefix) :]
    return f"{code_prefix}{track_folder}/{rel}.json"


def needs_new_summary_track(
    bucket: storage.Bucket,
    code_prefix: str,
    code_blob_name: str,
    track_folder: str,
) -> bool:
    sp = summary_blob_path_track(code_prefix, code_blob_name, track_folder)
    if not summary_json_exists(bucket, sp):
        return True
    rec = load_summary_record(bucket, sp)
    if not rec or rec.get("status") != "completed":
        return True
    if not rec.get("summary"):
        return True
    return False


def py_files_missing_summaries_track(
    bucket: storage.Bucket,
    code_prefix: str,
    py_blob_names: list[str],
    track_folder: str,
) -> list[str]:
    return [
        n
        for n in py_blob_names
        if needs_new_summary_track(bucket, code_prefix, n, track_folder)
    ]


def load_cached_results_for_kg_track(
    bucket: storage.Bucket,
    code_prefix: str,
    py_blob_names: list[str],
    track_folder: str,
) -> list[dict[str, Any]] | None:
    results: list[dict[str, Any]] = []
    for name in py_blob_names:
        sp = summary_blob_path_track(code_prefix, name, track_folder)
        rec = load_summary_record(bucket, sp)
        if not rec:
            return None
        summary = rec.get("summary")
        status = rec.get("status", "completed")
        if status != "completed" or not summary:
            logger.warning("Cached summary unusable for %s (track=%s)", name, track_folder)
            return None
        results.append(
            {
                "file": rec.get("file", name),
                "summary": str(summary),
                "status": "completed",
            }
        )
    return results


async def run_single_track(
    *,
    user_id: str,
    bucket: storage.Bucket,
    bucket_name: str,
    code_prefix: str,
    py_blob_names: list[str],
    model_name: str,
    max_parallel: int,
    track_key: str,
    summary_folder: str,
    summarizer_instruction: str,
    summary_focus_suffix: str,
    user_message_prefix: str | None,
    kg_batch_size: int,
    kg_batch_overlap: int,
    kg_max_doc_chars: int,
    extra_entity_props: dict[str, Any],
    neo4j_database: str | None = None,
) -> dict[str, Any]:
    """One summarization + KG build for a track."""
    adk_user = f"kg-pipeline3-{track_key}-{normalize_user_id(user_id)}"

    missing = py_files_missing_summaries_track(
        bucket, code_prefix, py_blob_names, summary_folder
    )

    if not missing:
        logger.info(
            "[%s] All %d file(s) cached — loading summaries only.",
            track_key,
            len(py_blob_names),
        )
        cached = load_cached_results_for_kg_track(
            bucket, code_prefix, py_blob_names, summary_folder
        )
        if cached is None:
            return {
                "track": track_key,
                "status": "error",
                "reason": "cached_summaries_unusable",
            }
        results = cached
    else:
        logger.info(
            "[%s] %d file(s) need new summaries (of %d).",
            track_key,
            len(missing),
            len(py_blob_names),
        )
        cached_by_file: dict[str, dict[str, Any]] = {}
        for name in py_blob_names:
            if name not in missing:
                rec = load_summary_record(
                    bucket, summary_blob_path_track(code_prefix, name, summary_folder)
                )
                if rec and rec.get("status") == "completed" and rec.get("summary"):
                    cached_by_file[name] = {
                        "file": rec.get("file", name),
                        "summary": str(rec["summary"]),
                        "status": "completed",
                    }

        to_generate = load_code_files_for_names(bucket, missing)
        if len(to_generate) != len(missing):
            return {
                "track": track_key,
                "status": "error",
                "reason": "code_load_failed",
            }

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

        for name, res in fresh_by_file.items():
            sp = summary_blob_path_track(code_prefix, name, summary_folder)
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
            logger.info("Wrote [%s] summary to gs://%s/%s", track_key, bucket_name, sp)

        results = merge_results_for_kg(py_blob_names, cached_by_file, fresh_by_file)

    logger.info("[%s] Building knowledge graph (%d summaries)...", track_key, len(results))
    kg_result = build_kg_and_push_to_neo4j(
        results,
        kg_batch_size=kg_batch_size,
        kg_batch_overlap=kg_batch_overlap,
        kg_max_doc_chars=kg_max_doc_chars,
        extra_entity_props=extra_entity_props,
        neo4j_database=neo4j_database,
    )
    logger.info("[%s] KG step: %s", track_key, kg_result)

    return {
        "track": track_key,
        "status": "completed",
        "summary_records": len(results),
        "kg_result": kg_result,
    }


async def run_pipeline3(
    user_id: str,
    *,
    neo4j_database: str | None = None,
) -> dict[str, Any]:
    """
    Run structural then technical track; both merge into the same Neo4j (:Entity / :REL)
    with ``kg_track_*`` flags on nodes for differentiation.

    If ``neo4j_database`` is set, writes go to that logical database (Neo4j 4+); otherwise
    the server default database is used.
    """
    bucket_name = Config.GCS_BUCKET_NAME
    code_prefix = code_prefix_for_user(user_id)
    model_name = Config.GEMINI_MODEL
    max_parallel = Config.SUMMARY_MAX_PARALLEL

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    logger.info(
        "pipeline3: gs://%s/%s (tracks: %s, %s)",
        bucket_name,
        code_prefix.rstrip("/"),
        SUMMARIES_STRUCTURAL,
        SUMMARIES_TECHNICAL,
    )

    py_blob_names = list_python_blob_names(bucket, code_prefix)
    if not py_blob_names:
        return {
            "status": "skipped",
            "reason": "no_py_files",
            "user_id": normalize_user_id(user_id),
        }

    structural = await run_single_track(
        user_id=user_id,
        bucket=bucket,
        bucket_name=bucket_name,
        code_prefix=code_prefix,
        py_blob_names=py_blob_names,
        model_name=model_name,
        max_parallel=max_parallel,
        track_key="structural",
        summary_folder=SUMMARIES_STRUCTURAL,
        summarizer_instruction=STRUCTURAL_SUMMARIZER_INSTRUCTION,
        summary_focus_suffix=STRUCTURAL_FOCUS_SUFFIX,
        user_message_prefix=None,
        kg_batch_size=STRUCTURAL_KG_BATCH,
        kg_batch_overlap=STRUCTURAL_KG_OVERLAP,
        kg_max_doc_chars=STRUCTURAL_MAX_DOC_CHARS,
        extra_entity_props={"kg_track_structural": True},
        neo4j_database=neo4j_database,
    )

    technical = await run_single_track(
        user_id=user_id,
        bucket=bucket,
        bucket_name=bucket_name,
        code_prefix=code_prefix,
        py_blob_names=py_blob_names,
        model_name=model_name,
        max_parallel=max_parallel,
        track_key="technical",
        summary_folder=SUMMARIES_TECHNICAL,
        summarizer_instruction=TECHNICAL_SUMMARIZER_INSTRUCTION,
        summary_focus_suffix=TECHNICAL_FOCUS_SUFFIX,
        user_message_prefix=TECHNICAL_USER_PREFIX,
        kg_batch_size=TECHNICAL_KG_BATCH,
        kg_batch_overlap=TECHNICAL_KG_OVERLAP,
        kg_max_doc_chars=TECHNICAL_MAX_DOC_CHARS,
        extra_entity_props={"kg_track_technical": True},
        neo4j_database=neo4j_database,
    )

    out: dict[str, Any] = {
        "status": "completed",
        "user_id": normalize_user_id(user_id),
        "py_file_count": len(py_blob_names),
        "structural": structural,
        "technical": technical,
    }
    if neo4j_database:
        out["neo4j_database"] = neo4j_database
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Dual-track KG: structural overview (batched, overlapping) + technical "
            "(code-grounded) graphs merged in Neo4j."
        )
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="GCS prefix, e.g. u_123",
    )
    parser.add_argument(
        "--neo4j-database",
        default=None,
        dest="neo4j_database",
        metavar="NAME",
        help=(
            "Neo4j logical database name to write to (Neo4j 4+, Enterprise). "
            "Omit to use the server default database."
        ),
    )
    args = parser.parse_args()
    db = (args.neo4j_database or "").strip() or None
    out = asyncio.run(run_pipeline3(args.user_id, neo4j_database=db))
    logger.info("pipeline3 finished: %s", out)


if __name__ == "__main__":
    main()
