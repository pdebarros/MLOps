"""
Load .py files from GCS under a user prefix, summarize with Gemini (ADK), persist
summaries to GCS, then build a KG (tools2 + Neo4j).

Summaries are stored at: gs://<bucket>/<user_id>/summaries/<relative-path>.py.json

If every .py file already has a summary object in GCS, summarization is skipped and
the pipeline loads cached summaries and proceeds to knowledge-graph creation.

Run from this directory:
  python pipeline2.py --user-id u_123
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import uuid
from typing import Any, List

from config import Config
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.cloud import storage
from google.genai import types

from tools2 import build_kg_and_push_to_neo4j

# Subfolder under <user_id>/ for cached summary JSON (code lives elsewhere under user_id/)
SUMMARIES_SEGMENT = "summaries"


class _TaskIdFilter(logging.Filter):
    """Ensure %(task_id)s exists for Formatter (unused on non-adapter logs)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not getattr(record, "task_id", None):
            record.task_id = "-"
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(task_id)s] %(message)s",
)
_task_id_filter = _TaskIdFilter()
for _hdlr in logging.root.handlers:
    _hdlr.addFilter(_task_id_filter)
logger = logging.getLogger("Orchestrator")


def _log_adapter(task_id: str) -> logging.LoggerAdapter:
    return logging.LoggerAdapter(logger, {"task_id": task_id})


def normalize_user_id(user_id: str) -> str:
    uid = user_id.strip().strip("/")
    if not uid:
        raise ValueError("user_id must be non-empty")
    return uid


def code_prefix_for_user(user_id: str) -> str:
    """Prefix for all blobs belonging to this user (e.g. u_123/)."""
    return f"{normalize_user_id(user_id)}/"


def summary_blob_path(code_prefix: str, code_blob_name: str) -> str:
    """
    Map gs://bucket/u_123/foo/bar.py -> gs://bucket/u_123/summaries/foo/bar.py.json
    """
    if not code_blob_name.startswith(code_prefix):
        raise ValueError(f"code blob {code_blob_name!r} must start with {code_prefix!r}")
    rel = code_blob_name[len(code_prefix) :]
    return f"{code_prefix}{SUMMARIES_SEGMENT}/{rel}.json"


def list_python_blob_names(bucket: storage.Bucket, code_prefix: str) -> list[str]:
    """List .py object names under code_prefix, excluding the summaries/ tree."""
    out: list[str] = []
    summaries_guard = f"{code_prefix}{SUMMARIES_SEGMENT}/"
    for blob in bucket.list_blobs(prefix=code_prefix):
        name = blob.name
        if name.startswith(summaries_guard):
            continue
        if name.endswith(".py"):
            out.append(name)
    return sorted(out)


def summary_json_exists(bucket: storage.Bucket, summary_path: str) -> bool:
    return bucket.blob(summary_path).exists()


def load_summary_record(bucket: storage.Bucket, summary_path: str) -> dict[str, Any] | None:
    try:
        b = bucket.blob(summary_path)
        if not b.exists():
            return None
        return json.loads(b.download_as_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load summary %s: %s", summary_path, e)
        return None


def upload_summary_record(
    bucket: storage.Bucket, summary_path: str, record: dict[str, Any]
) -> None:
    body = json.dumps(record, ensure_ascii=False, indent=2)
    blob = bucket.blob(summary_path)
    blob.upload_from_string(body, content_type="application/json; charset=utf-8")


def load_code_files_for_names(
    bucket: storage.Bucket, blob_names: list[str]
) -> list[dict[str, str]]:
    """Download listed blobs as file_name / file_content."""
    file_data_list: list[dict[str, str]] = []
    for name in blob_names:
        try:
            blob = bucket.blob(name)
            content_text = blob.download_as_bytes().decode("utf-8")
            file_data_list.append({"file_name": name, "file_content": content_text})
            logger.info("Loaded code: %s", name)
        except Exception as e:
            logger.error("Error loading %s: %s", name, e)
    return file_data_list


def load_code_files(
    bucket_name: str,
    prefix: str | None = None,
) -> List[dict[str, str]]:
    """List .py blobs in GCS and return dicts with file_name and file_content."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if not prefix:
        return []
    names = list_python_blob_names(bucket, prefix)
    return load_code_files_for_names(bucket, names)


def needs_new_summary(bucket: storage.Bucket, code_prefix: str, code_blob_name: str) -> bool:
    """True if there is no usable completed summary JSON in GCS for this .py file."""
    sp = summary_blob_path(code_prefix, code_blob_name)
    if not summary_json_exists(bucket, sp):
        return True
    rec = load_summary_record(bucket, sp)
    if not rec or rec.get("status") != "completed":
        return True
    if not rec.get("summary"):
        return True
    return False


def py_files_missing_summaries(
    bucket: storage.Bucket, code_prefix: str, py_blob_names: list[str]
) -> list[str]:
    """Return .py blob names that need summarization (missing or failed cache)."""
    return [n for n in py_blob_names if needs_new_summary(bucket, code_prefix, n)]


def load_cached_results_for_kg(
    bucket: storage.Bucket, code_prefix: str, py_blob_names: list[str]
) -> list[dict[str, Any]] | None:
    """
    Load summary JSON for each path. Returns None if any record is missing or invalid.
    """
    results: list[dict[str, Any]] = []
    for name in py_blob_names:
        sp = summary_blob_path(code_prefix, name)
        rec = load_summary_record(bucket, sp)
        if not rec:
            return None
        summary = rec.get("summary")
        status = rec.get("status", "completed")
        if status != "completed" or not summary:
            logger.warning("Cached summary unusable for %s (status=%s)", name, status)
            return None
        results.append(
            {
                "file": rec.get("file", name),
                "summary": str(summary),
                "status": "completed",
            }
        )
    return results


async def _final_text_from_events(events_source) -> str | None:
    """Collect the last non-user final response text from a runner stream."""
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


async def summarize_one_file(
    runner: Runner,
    semaphore: asyncio.Semaphore,
    file_name: str,
    file_content: str,
    adk_user: str,
    *,
    user_message_prefix: str | None = None,
    focus_suffix: str = "",
) -> dict[str, Any]:
    """Run the summarizer agent for one file via ADK Runner."""
    task_id = str(uuid.uuid4())[:8]
    log = _log_adapter(task_id)
    log.info("START: Analyzing '%s'", file_name)
    async with semaphore:
        try:
            session_id = f"sum-{uuid.uuid4().hex}"
            base_intro = (
                user_message_prefix
                if user_message_prefix is not None
                else (
                    f"File: {file_name}\n\n"
                    "Analyze this Python code. Extract classes, "
                    "dependencies, and logic flow.\n\n"
                )
            )
            if "{file_name}" in base_intro or "{file_content}" in base_intro:
                body = base_intro.format(file_name=file_name, file_content=file_content)
            else:
                body = f"{base_intro}{file_content}"
            if focus_suffix:
                body = f"{body}\n{focus_suffix}"
            msg = types.Content(
                role="user",
                parts=[
                    types.Part(
                        text=body,
                    )
                ],
            )
            text = await _final_text_from_events(
                runner.run_async(
                    user_id=adk_user,
                    session_id=session_id,
                    new_message=msg,
                )
            )
            if not text:
                raise RuntimeError("No model text in response stream")
            log.info("SUCCESS: Summary generated for '%s'", file_name)
            return {
                "file": file_name,
                "summary": text,
                "status": "completed",
            }
        except Exception as e:
            log.error("FAILURE: '%s' failed. Error: %s", file_name, e)
            return {
                "file": file_name,
                "summary": None,
                "status": "failed",
                "error": str(e),
            }


DEFAULT_SUMMARIZER_INSTRUCTION = (
    "You are a code analyst. Extract classes, dependencies, "
    "and logic flow from Python source. Respond with a clear, "
    "structured summary."
)


async def run_parallel_summaries(
    files: List[dict[str, str]],
    model_name: str,
    max_parallel: int,
    adk_user: str,
    *,
    summarizer_instruction: str | None = None,
    user_message_prefix: str | None = None,
    summary_focus_suffix: str = "",
) -> List[dict[str, Any]]:
    summarizer_agent = Agent(
        name="code_summarizer",
        model=model_name,
        instruction=(summarizer_instruction or DEFAULT_SUMMARIZER_INSTRUCTION),
    )
    runner = Runner(
        app_name="kg-pipeline",
        agent=summarizer_agent,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )
    sem = asyncio.Semaphore(max_parallel)
    tasks = [
        summarize_one_file(
            runner,
            sem,
            item["file_name"],
            item["file_content"],
            adk_user,
            user_message_prefix=user_message_prefix,
            focus_suffix=summary_focus_suffix,
        )
        for item in files
    ]
    return await asyncio.gather(*tasks)


def merge_results_for_kg(
    py_order: list[str],
    cached_by_file: dict[str, dict[str, Any]],
    fresh_by_file: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Stable order matching py_order; prefer fresh over cached."""
    out: list[dict[str, Any]] = []
    for path in py_order:
        if path in fresh_by_file:
            out.append(fresh_by_file[path])
        elif path in cached_by_file:
            out.append(cached_by_file[path])
        else:
            logger.error("No result for %s during merge", path)
            out.append({"file": path, "summary": None, "status": "failed"})
    return out


async def run_pipeline(user_id: str) -> dict[str, Any] | None:
    bucket_name = Config.GCS_BUCKET_NAME
    code_prefix = code_prefix_for_user(user_id)
    model_name = Config.GEMINI_MODEL
    max_parallel = Config.SUMMARY_MAX_PARALLEL
    adk_user = f"kg-pipeline-{normalize_user_id(user_id)}"

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    logger.info(
        "User prefix gs://%s/%s (summaries under .../%s/)",
        bucket_name,
        code_prefix.rstrip("/"),
        SUMMARIES_SEGMENT,
    )

    py_blob_names = list_python_blob_names(bucket, code_prefix)
    if not py_blob_names:
        logger.warning("No .py files under gs://%s/%s — exiting.", bucket_name, code_prefix)
        return {"status": "skipped", "reason": "no_py_files", "user_id": normalize_user_id(user_id)}

    missing = py_files_missing_summaries(bucket, code_prefix, py_blob_names)

    if not missing:
        logger.info(
            "All %d .py file(s) already have summaries in GCS — skipping summarization.",
            len(py_blob_names),
        )
        cached = load_cached_results_for_kg(bucket, code_prefix, py_blob_names)
        if cached is None:
            logger.error("Could not load cached summaries; aborting KG step.")
            return {"status": "error", "reason": "cached_summaries_unusable", "user_id": normalize_user_id(user_id)}
        results = cached
    else:
        logger.info(
            "%d .py file(s) need new summaries (out of %d total).",
            len(missing),
            len(py_blob_names),
        )
        cached_by_file: dict[str, dict[str, Any]] = {}
        for name in py_blob_names:
            if name not in missing:
                rec = load_summary_record(
                    bucket, summary_blob_path(code_prefix, name)
                )
                if rec and rec.get("status") == "completed" and rec.get("summary"):
                    cached_by_file[name] = {
                        "file": rec.get("file", name),
                        "summary": str(rec["summary"]),
                        "status": "completed",
                    }

        to_generate = load_code_files_for_names(bucket, missing)
        if len(to_generate) != len(missing):
            logger.error("Failed to load some code blobs for summarization.")
            return {"status": "error", "reason": "code_load_failed", "user_id": normalize_user_id(user_id)}

        fresh_list = await run_parallel_summaries(
            to_generate, model_name, max_parallel, adk_user
        )
        fresh_by_file = {r["file"]: r for r in fresh_list}

        for name, res in fresh_by_file.items():
            sp = summary_blob_path(code_prefix, name)
            upload_summary_record(
                bucket,
                sp,
                {
                    "file": name,
                    "summary": res.get("summary"),
                    "status": res.get("status"),
                    "error": res.get("error"),
                },
            )
            logger.info("Wrote summary to gs://%s/%s", bucket_name, sp)

        results = merge_results_for_kg(py_blob_names, cached_by_file, fresh_by_file)

    logger.info("Building knowledge graph from %d summary record(s)...", len(results))
    kg_result = build_kg_and_push_to_neo4j(results)
    logger.info("KG step result: %s", kg_result)
    return {
        "status": "completed",
        "user_id": normalize_user_id(user_id),
        "py_file_count": len(py_blob_names),
        "summary_records": len(results),
        "kg_result": kg_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Python under gs://<bucket>/<user_id>/ and build a KG. "
            "Caches per-file summaries under <user_id>/summaries/."
        )
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help='GCS extension folder, e.g. u_123 → gs://codebases-03-26/u_123/',
    )
    args = parser.parse_args()
    asyncio.run(run_pipeline(args.user_id))


if __name__ == "__main__":
    main()
