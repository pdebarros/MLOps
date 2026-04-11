"""
Gemini Flash (Vertex AI) multi-step Cypher explorer: iterates read-only queries until it
emits a final natural-language answer grounded in accumulated Neo4j results.

Uses the same Neo4j schema fetch, read-only guard, and property-key lint as ``model.py``.
Intended for experiments alongside the Groq weak student; wire into tools/agent separately.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

from .config import config, google_cloud_project, project_and_location_for_vertex
from .model import (
    NOT_FOUND_MESSAGE,
    _ensure_non_empty_response,
    _extract_cypher_block,
    _lint_cypher_property_keys,
    _neo4j_driver,
    _neo4j_property_allowlist,
    _run_cypher_read,
    _validate_read_only_cypher,
    fetch_neo4j_schema_text,
)

logger = logging.getLogger("eval_agent.model2")

_STEP_PREFIX = "GEMINI_CYPHER_STEP"
_TRACE_PREFIX = "GEMINI_CYPHER_TRACE"

_gemini_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini_cypher")

_LLM: ChatVertexAI | None = None

# Defaults (override with env)
_GEMINI_MODEL = os.environ.get("GEMINI_CYPHER_MODEL", "gemini-2.5-flash")
_MAX_STEPS = max(1, int(os.environ.get("GEMINI_CYPHER_MAX_STEPS", "5")))
_MAX_OBS_JSON_CHARS = max(2000, int(os.environ.get("GEMINI_CYPHER_MAX_OBS_CHARS", "12000")))
_GEMINI_TEMPERATURE = float(os.environ.get("GEMINI_CYPHER_TEMPERATURE", "0.2"))


def _trunc_trace_text(s: str | None) -> str:
    cap = max(256, int(config.GEMINI_CYPHER_TRACE_MAX_TEXT))
    if s is None:
        return ""
    s = str(s)
    if len(s) <= cap:
        return s
    return s[:cap] + "…"


def _emit_gemini_session_trace(record: dict[str, Any]) -> None:
    """One JSON summary per `gemini_flash_kg_query` run; optional JSONL append."""
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    record.setdefault("trace", "gemini_cypher_v1")
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except TypeError:
        line = json.dumps({k: str(v) for k, v in record.items()}, ensure_ascii=False)
    logger.info("%s %s", _TRACE_PREFIX, line)
    path = (config.GEMINI_CYPHER_TRACE_LOG or "").strip()
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        logger.warning("GEMINI_CYPHER_TRACE_LOG write failed (%s): %s", path, e)


def _log_gemini_step(session_id: str, step_record: dict[str, Any]) -> None:
    """Per-step line for tailing logs during a multi-step run."""
    payload = {"session_id": session_id, **step_record}
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        line = json.dumps({k: str(v) for k, v in payload.items()}, ensure_ascii=False)
    logger.info("%s %s", _STEP_PREFIX, line)


def _run_in_gemini_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
    fut = _gemini_executor.submit(fn, *args, **kwargs)
    return fut.result()


def _vertex_project_and_location() -> tuple[str | None, str]:
    corpus = config.VERTEX_RAG_CORPUS
    if corpus:
        p, loc = project_and_location_for_vertex(corpus)
        return p, loc
    loc = (
        os.environ.get("VERTEX_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_REGION")
        or "us-central1"
    )
    return google_cloud_project(), loc


def _get_chat_vertex() -> ChatVertexAI:
    global _LLM
    if _LLM is not None:
        return _LLM
    project, location = _vertex_project_and_location()
    kwargs: dict[str, Any] = {
        "model": _GEMINI_MODEL,
        "location": location,
        "temperature": _GEMINI_TEMPERATURE,
    }
    if project:
        kwargs["project"] = project
    logger.info(
        "Gemini Cypher explorer LLM (model=%s, location=%s, project=%s)",
        _GEMINI_MODEL,
        location,
        project or "(ADC default)",
    )
    _LLM = ChatVertexAI(**kwargs)
    return _LLM


def _parse_action_json(text: str) -> dict[str, Any] | None:
    """Parse a single JSON object from the model; tolerate ```json fences."""
    t = (text or "").strip()
    if not t:
        return None
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Brace slice fallback
    i = t.find("{")
    j = t.rfind("}")
    if i >= 0 and j > i:
        try:
            obj = json.loads(t[i : j + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _build_system_prompt(
    schema_text: str,
    eval_agent_prompt: str | None,
) -> str:
    parts = [
        "You explore a Neo4j 5.x knowledge graph with **read-only** Cypher. "
        "You may run multiple queries; after each query you receive JSON with rows or an error. "
        "Answer only from observations — no outside knowledge.",
        "",
        "Reply with **only** a JSON object (no markdown outside the JSON), one of:",
        '- {"action":"query","cypher":"<single read-only Cypher statement>"}',
        '- {"action":"final","answer":"<concise answer, or exactly: Information not found.>"}',
        "",
        "Rules for Cypher:",
        "- One statement only; no semicolons chaining writes.",
        "- Use MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN / ORDER BY / LIMIT / SKIP / UNWIND / COLLECT only.",
        "- Prefer case-insensitive text: toLower(...), CONTAINS.",
        "- Use only property keys that exist in the schema below.",
        "",
        "## Graph schema",
        schema_text,
    ]
    if eval_agent_prompt:
        parts.extend(["", "## Evaluator / workflow context (from eval agent)", eval_agent_prompt.strip()])
    return "\n".join(parts)


def _truncate_observation(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(raw) <= _MAX_OBS_JSON_CHARS:
        return raw
    return raw[:_MAX_OBS_JSON_CHARS] + "\n… (truncated)"


def _gemini_multi_step_sync(
    question: str,
    user_id: str,
    eval_agent_prompt: str | None = None,
) -> str:
    session_id = str(uuid.uuid4())
    session_steps: list[dict[str, Any]] = []
    database = (user_id or "").strip()
    q_preview = _trunc_trace_text(question.strip())

    def _finish(outcome: str, returned: str, extra: dict[str, Any] | None = None) -> str:
        rec: dict[str, Any] = {
            "session_id": session_id,
            "neo4j_database": database or None,
            "gemini_model": _GEMINI_MODEL,
            "question_preview": q_preview,
            "has_eval_agent_prompt": bool((eval_agent_prompt or "").strip()),
            "steps": session_steps,
            "final_outcome": outcome,
            "returned_preview": _trunc_trace_text(returned),
        }
        if extra:
            rec.update(extra)
        _emit_gemini_session_trace(rec)
        return returned

    if not database:
        return _finish("no_database", NOT_FOUND_MESSAGE)

    schema_text = fetch_neo4j_schema_text(database)
    prop_allow = _neo4j_property_allowlist(database)
    system = _build_system_prompt(schema_text, eval_agent_prompt)
    llm = _get_chat_vertex()

    messages: list[Any] = [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Question:\n{question.strip()}\n\n"
                "Begin: output one JSON object. Prefer a `query` first unless the schema already "
                "implies the question is unanswerable."
            )
        ),
    ]

    for step in range(_MAX_STEPS):
        try:
            resp = llm.invoke(messages)
            raw = (resp.content or "").strip()
        except Exception as e:
            logger.exception("Gemini invoke failed")
            session_steps.append(
                {
                    "loop_index": step + 1,
                    "kind": "gemini_invoke",
                    "ok": False,
                    "error": str(e),
                }
            )
            _log_gemini_step(session_id, session_steps[-1])
            err = f"gemini_flash_kg_query error: {e}"
            return _finish("gemini_invoke_error", err, {"error": str(e)})

        session_steps.append(
            {
                "loop_index": step + 1,
                "kind": "gemini_turn",
                "raw_reply": _trunc_trace_text(raw),
            }
        )
        data = _parse_action_json(raw)
        if not data:
            session_steps[-1]["parse_ok"] = False
            _log_gemini_step(session_id, session_steps[-1])
            messages.append(AIMessage(content=raw))
            messages.append(
                HumanMessage(
                    content=(
                        "Invalid or non-JSON reply. Respond with **only** "
                        '{"action":"query","cypher":"..."} or {"action":"final","answer":"..."}'
                    )
                )
            )
            continue

        session_steps[-1]["parse_ok"] = True
        session_steps[-1]["parsed"] = {
            k: _trunc_trace_text(v) if isinstance(v, str) else v
            for k, v in data.items()
        }
        _log_gemini_step(session_id, session_steps[-1])

        action = (data.get("action") or "").strip().lower()
        if action == "final":
            ans = data.get("answer")
            final = _ensure_non_empty_response(
                ans if isinstance(ans, str) else str(ans)
            )
            outcome = "not_found" if final == NOT_FOUND_MESSAGE else "success"
            session_steps.append(
                {
                    "loop_index": step + 1,
                    "kind": "final_answer",
                    "answer_preview": _trunc_trace_text(final),
                }
            )
            _log_gemini_step(session_id, session_steps[-1])
            return _finish(outcome, final)

        if action != "query":
            session_steps.append(
                {
                    "loop_index": step + 1,
                    "kind": "unknown_action",
                    "action": action,
                }
            )
            _log_gemini_step(session_id, session_steps[-1])
            messages.append(AIMessage(content=raw))
            messages.append(
                HumanMessage(
                    content='Unknown action. Use "query" or "final" only in the JSON "action" field.'
                )
            )
            continue

        cypher = data.get("cypher")
        if not isinstance(cypher, str) or not cypher.strip():
            alt = _extract_cypher_block(raw)
            cypher = alt or ""
        cypher = (cypher or "").strip()
        if not cypher:
            session_steps.append(
                {
                    "loop_index": step + 1,
                    "kind": "missing_cypher",
                }
            )
            _log_gemini_step(session_id, session_steps[-1])
            messages.append(AIMessage(content=raw))
            messages.append(
                HumanMessage(
                    content="Missing cypher. Provide a non-empty string in the `cypher` field."
                )
            )
            continue

        ok, reason = _validate_read_only_cypher(cypher)
        if not ok:
            obs = {"step": step + 1, "ok": False, "phase": "validate", "error": reason}
            session_steps.append(
                {
                    "loop_index": step + 1,
                    "kind": "validate_cypher",
                    "ok": False,
                    "reason": reason,
                    "cypher": _trunc_trace_text(cypher),
                }
            )
            _log_gemini_step(session_id, session_steps[-1])
            messages.append(AIMessage(content=raw))
            messages.append(
                HumanMessage(
                    content=f"Observation:\n{_truncate_observation(obs)}\n\nRevise the query or answer final."
                )
            )
            continue

        lint_ok, lint_reason = _lint_cypher_property_keys(cypher, prop_allow)
        if not lint_ok:
            obs = {
                "step": step + 1,
                "ok": False,
                "phase": "property_lint",
                "error": lint_reason,
            }
            session_steps.append(
                {
                    "loop_index": step + 1,
                    "kind": "schema_property_lint",
                    "ok": False,
                    "reason": lint_reason,
                    "cypher": _trunc_trace_text(cypher),
                }
            )
            _log_gemini_step(session_id, session_steps[-1])
            messages.append(AIMessage(content=raw))
            messages.append(
                HumanMessage(
                    content=f"Observation:\n{_truncate_observation(obs)}\n\nRevise the query or answer final."
                )
            )
            continue

        driver = _neo4j_driver()
        try:
            with driver.session(database=database) as session:
                rows, err = _run_cypher_read(session, cypher)
        finally:
            driver.close()

        obs = {
            "step": step + 1,
            "ok": err is None,
            "phase": "neo4j",
            "row_count": len(rows),
            "error": err,
            "rows": rows,
        }
        # Compact row preview for trace (avoid huge JSON in logs)
        max_rows_preview = 15
        rows_for_log: list[Any] | dict[str, Any] = rows[:max_rows_preview]
        if len(rows) > max_rows_preview:
            rows_for_log = {
                "_truncated": True,
                "row_count": len(rows),
                "first_rows": rows[:max_rows_preview],
            }

        session_steps.append(
            {
                "loop_index": step + 1,
                "kind": "neo4j_execute",
                "cypher_executed": _trunc_trace_text(cypher),
                "row_count": len(rows),
                "error": err,
                "rows_preview": rows_for_log,
            }
        )
        _log_gemini_step(session_id, session_steps[-1])

        messages.append(AIMessage(content=raw))
        messages.append(
            HumanMessage(
                content=(
                    f"Observation:\n{_truncate_observation(obs)}\n\n"
                    "If these rows (or empty result with no error) are enough, respond with "
                    '{"action":"final","answer":"..."}. Otherwise issue another '
                    '{"action":"query","cypher":"..."}.'
                )
            )
        )

    return _finish("max_steps_exhausted", NOT_FOUND_MESSAGE)


def gemini_flash_kg_query(
    question: str,
    user_id: str,
    *,
    eval_agent_prompt: str | None = None,
) -> str:
    """
    Run Gemini Flash on Vertex with multi-step read-only Cypher exploration.

    No RAG or external grounding text is injected — only the graph schema (and optional
    ``eval_agent_prompt`` if you pass it from Python directly).

    Parameters
    ----------
    question:
        Natural-language question (same style as ``weak_model_query``).
    user_id:
        Neo4j database name (same as ``weak_model_query``'s ``user_id``).
    eval_agent_prompt:
        Optional full or partial **eval agent** instructions (e.g. import ``SYSTEM_INSTRUCTION``
        from ``eval_agent.prompt`` and pass it here). The ADK tool path does not pass this.
    """
    return _run_in_gemini_thread(
        _gemini_multi_step_sync,
        question,
        user_id,
        eval_agent_prompt,
    )


def gemini_flash_kg_query_with_eval_prompt(
    question: str,
    user_id: str,
) -> str:
    """Convenience wrapper that passes ``SYSTEM_INSTRUCTION`` from ``eval_agent.prompt``."""
    from .prompt import SYSTEM_INSTRUCTION

    return gemini_flash_kg_query(
        question,
        user_id,
        eval_agent_prompt=SYSTEM_INSTRUCTION,
    )
