"""
Weak student: Groq (Llama 8B) proposes read-only Cypher; results are executed on Neo4j and
fed back for a grounded answer. No LlamaIndex / PropertyGraphIndex.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from openai import OpenAI

from .config import config

logger = logging.getLogger("eval_agent.model")

_query_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weak_cypher")

_TRACE_PREFIX = "WEAK_CYPHER_TRACE"


def _trunc_text(s: str | None, max_len: int | None = None) -> str:
    if s is None:
        return ""
    cap = max_len if max_len is not None else int(config.WEAK_CYPHER_TRACE_MAX_TEXT)
    cap = max(256, cap)
    s = str(s)
    if len(s) <= cap:
        return s
    return s[:cap] + "…"


def _emit_weak_cypher_trace(record: dict[str, Any]) -> None:
    """Log one structured trace line (and optionally append JSONL to disk)."""
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    record.setdefault("trace", "weak_cypher_v1")
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except TypeError:
        line = json.dumps({k: str(v) for k, v in record.items()}, ensure_ascii=False)
    logger.info("%s %s", _TRACE_PREFIX, line)
    path = (config.WEAK_CYPHER_TRACE_LOG or "").strip()
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        logger.warning("WEAK_CYPHER_TRACE_LOG write failed (%s): %s", path, e)

NOT_FOUND_MESSAGE = "Information not found."

_schema_cache: dict[str, tuple[str, frozenset[str]]] = {}

# Write / admin operations must never run from generated Cypher.
_CYPHER_FORBIDDEN = re.compile(
    r"\b(?:CREATE|MERGE|DELETE|DETACH\s+DELETE|REMOVE|SET|DROP|"
    r"LOAD\s+CSV|FOREACH|GRANT|DENY|REVOKE|ALTER|START\s+DATABASE|STOP\s+DATABASE)\b",
    re.IGNORECASE,
)

# `variable.property` — excludes `1.0` floats (must start with letter or underscore).
_CYPHER_VAR_DOT_PROP = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b"
)


def _ensure_non_empty_response(raw: str | None) -> str:
    if raw is None:
        return NOT_FOUND_MESSAGE
    text = str(raw).strip()
    if not text:
        return NOT_FOUND_MESSAGE
    if text.lower() == "empty response":
        return NOT_FOUND_MESSAGE
    return text


def _run_in_query_thread(fn: Any, *args: Any, **kwargs: Any) -> Any:
    fut = _query_executor.submit(fn, *args, **kwargs)
    return fut.result()


def _neo4j_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


def _load_neo4j_schema_bundle(database: str) -> tuple[str, frozenset[str]]:
    """Schema markdown for prompts + property-key allowlist for pre-execution lint."""
    if database in _schema_cache:
        return _schema_cache[database]
    lines: list[str] = []
    prop_keys: set[str] = set()
    try:
        driver = _neo4j_driver()
        try:
            with driver.session(database=database) as session:
                try:
                    pk_row = session.run(
                        "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys"
                    ).single()
                    raw_keys = pk_row["keys"] if pk_row else []
                    prop_keys = {str(k) for k in raw_keys if k is not None and str(k)}
                except Exception as e:
                    logger.info("db.propertyKeys unavailable: %s", e)

                row = session.run(
                    "CALL db.labels() YIELD label RETURN collect(label) AS labels"
                ).single()
                labels = row["labels"] if row else []
                lines.append("## Node labels\n" + ", ".join(sorted(map(str, labels))))

                row = session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType "
                    "RETURN collect(relationshipType) AS types"
                ).single()
                rtypes = row["types"] if row else []
                lines.append("## Relationship types\n" + ", ".join(sorted(map(str, rtypes))))

                try:
                    q = """
                    CALL db.schema.nodeTypeProperties()
                    YIELD nodeType, propertyName, propertyTypes
                    RETURN nodeType, propertyName, propertyTypes
                    LIMIT 200
                    """
                    recs = session.run(q)
                    lines.append("## Node properties (sample)")
                    for i, r in enumerate(recs):
                        if i >= 80:
                            lines.append("… (truncated)")
                            break
                        lines.append(
                            f"- `{r['nodeType']}` · `{r['propertyName']}` : {r['propertyTypes']}"
                        )
                except Exception as e:
                    logger.info("nodeTypeProperties unavailable: %s", e)
                    lines.append("## Node properties\n(catalog call unavailable)")

                try:
                    q = """
                    CALL db.schema.relTypeProperties()
                    YIELD relType, propertyName, propertyTypes
                    RETURN relType, propertyName, propertyTypes
                    LIMIT 200
                    """
                    recs = session.run(q)
                    lines.append("## Relationship properties (sample)")
                    for i, r in enumerate(recs):
                        if i >= 80:
                            lines.append("… (truncated)")
                            break
                        lines.append(
                            f"- `{r['relType']}` · `{r['propertyName']}` : {r['propertyTypes']}"
                        )
                except Exception as e:
                    logger.info("relTypeProperties unavailable: %s", e)
                    lines.append("## Relationship properties\n(catalog call unavailable)")
        finally:
            driver.close()
    except Exception as e:
        logger.exception("fetch_neo4j_schema_text failed")
        lines.append(f"(schema fetch error: {e})")

    text = "\n".join(lines)
    bundle = (text, frozenset(prop_keys))
    _schema_cache[database] = bundle
    return bundle


def fetch_neo4j_schema_text(database: str) -> str:
    """Introspect labels, relationship types, and property catalogs (read-only)."""
    text, _keys = _load_neo4j_schema_bundle(database)
    return text


def _neo4j_property_allowlist(database: str) -> frozenset[str]:
    _text, keys = _load_neo4j_schema_bundle(database)
    return keys


def _strip_cypher_comments_and_strings(cypher: str) -> str:
    """Remove // and /* */ comments and string literals so lint ignores text inside them."""
    s = re.sub(r"//[^\n]*", "", cypher)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)
    s = re.sub(r"'(?:[^'\\]|\\.)*'", "''", s)
    s = re.sub(r'"(?:[^"\\]|\\.)*"', '""', s)
    return s


def _lint_cypher_property_keys(
    cypher: str, allowed: frozenset[str]
) -> tuple[bool, str]:
    """
    Reject queries that reference property keys not present in the DB catalog
    (avoids UnknownPropertyKeyWarning and wasted execution).
    If ``allowed`` is empty, lint is skipped (unknown / empty graph).
    """
    if not allowed:
        return True, ""
    s = _strip_cypher_comments_and_strings(cypher)
    unknown: set[str] = set()
    for m in _CYPHER_VAR_DOT_PROP.finditer(s):
        prop = m.group(2)
        if prop not in allowed:
            unknown.add(prop)
    if unknown:
        names = ", ".join(sorted(unknown))
        return False, f"unknown property key(s) not in database: {names}"
    return True, ""


def _validate_read_only_cypher(cypher: str) -> tuple[bool, str]:
    s = cypher.strip()
    if not s:
        return False, "empty cypher"
    if ";" in s.rstrip().rstrip(";"):
        parts = [p.strip() for p in s.split(";") if p.strip()]
        if len(parts) > 1:
            return False, "multiple statements are not allowed"
    if _CYPHER_FORBIDDEN.search(s):
        return False, "forbidden keyword in Cypher (read-only only)"
    return True, ""


def _extract_cypher_block(text: str) -> str | None:
    m = re.search(r"```(?:cypher)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    t = text.strip()
    if t.upper().startswith("MATCH") or t.upper().startswith("OPTIONAL MATCH"):
        return t
    return None


def _run_cypher_read(session, cypher: str) -> tuple[list[dict[str, Any]], str | None]:
    ok, reason = _validate_read_only_cypher(cypher)
    if not ok:
        return [], reason
    rows: list[dict[str, Any]] = []
    try:
        result = session.run(cypher)
        cap = max(1, int(config.WEAK_CYPHER_MAX_ROWS))
        for i, record in enumerate(result):
            if i >= cap:
                break
            rows.append(dict(record))
    except Exception as e:
        logger.info("Cypher execution failed: %s", e)
        return [], str(e)
    return rows, None


def _groq_chat(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Returns (assistant_text, groq_response_meta) for tracing."""
    key = (config.GROQ_API_KEY or "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY (or GROQ_KEY) is not set. Add it to eval_agent/.env."
        )
    m = model or config.GROQ_MODEL
    client = OpenAI(api_key=key, base_url=config.GROQ_BASE_URL)
    resp = client.chat.completions.create(
        model=m,
        messages=messages,
        temperature=0.1,
    )
    ch0 = resp.choices[0]
    choice = ch0.message
    text = (choice.content or "").strip()
    usage_obj = getattr(resp, "usage", None)
    usage: dict[str, Any] | None = None
    if usage_obj is not None:
        usage = {
            "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
            "completion_tokens": getattr(usage_obj, "completion_tokens", None),
            "total_tokens": getattr(usage_obj, "total_tokens", None),
        }
    meta: dict[str, Any] = {
        "groq_response_id": getattr(resp, "id", None),
        "model": getattr(resp, "model", None) or m,
        "usage": usage,
        "finish_reason": getattr(ch0, "finish_reason", None),
    }
    return text, meta


def _resolve_llm_model(llm: Any | None) -> str | None:
    if llm is None:
        return None
    for attr in ("model", "model_name", "_model"):
        v = getattr(llm, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _cypher_student_sync(
    question: str,
    user_id: str,
    extra_context: str | None = None,
    llm: Any | None = None,
) -> str:
    cap = int(config.WEAK_CYPHER_TRACE_MAX_TEXT)
    database = (user_id or "").strip()
    q_preview = _trunc_text(question.strip(), cap)
    model_name = _resolve_llm_model(llm) or config.GROQ_MODEL

    def _finish(
        outcome: str,
        returned: str,
        steps: list[dict[str, Any]],
        extra: dict[str, Any] | None = None,
    ) -> str:
        rec: dict[str, Any] = {
            "neo4j_database": database or None,
            "groq_model": model_name,
            "question_preview": q_preview,
            "steps": steps,
            "final_outcome": outcome,
            "returned_preview": _trunc_text(returned, cap),
        }
        if extra:
            rec.update(extra)
        _emit_weak_cypher_trace(rec)
        return returned

    if not database:
        return _finish("no_database", NOT_FOUND_MESSAGE, [])

    schema_text = fetch_neo4j_schema_text(database)

    gen_system = (
        "You generate exactly ONE read-only Cypher query for Neo4j 5.x against the knowledge graph. "
        "The database may use multi-database mode; only query graph data (MATCH …), not admin commands.\n\n"
        "Rules:\n"
        "- Output ONLY a single fenced block: ```cypher\\n...\\n``` and nothing else after it.\n"
        "- Use only MATCH / OPTIONAL MATCH / WITH / WHERE / RETURN / ORDER BY / LIMIT / SKIP / UNWIND / COLLECT.\n"
        "- Do NOT use CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD CSV, FOREACH, or any schema/admin commands.\n"
        "- For user-provided text/entity matching, prefer case-insensitive matching with `toLower(...)`.\n"
        "- Prefer `CONTAINS` for text matching over strict equality when possible.\n"
        "- Example text filter: `WHERE toLower(coalesce(n.id, '')) CONTAINS toLower('gru')`.\n"
        "- When exact equality is necessary, still keep it case-insensitive, e.g. `toLower(n.id) = toLower('GRUModel')`.\n"
        "- Prefer limiting rows (e.g. LIMIT 50).\n"
        "- Use Neo4j syntax: integer ranges use `range(1, 5)` with `UNWIND range(...) AS x`, not `[1..5]`.\n"
        "- Pattern syntax: `(a:Label)-[:REL]->(b)`; keep parentheses balanced.\n"
        "- The KG pipeline uses :Entity nodes and :REL relationships; entity ids are often on `id`.\n"
        "- Only use property names that exist in this database (see catalog below); do not invent `name`, "
        "`description`, etc. unless listed.\n\n"
        f"## Graph schema (authoritative)\n{schema_text}\n"
    )
    if extra_context:
        gen_system += f"\n## Extra context\n{extra_context}\n"

    gen_user = (
        f"Question:\n{question.strip()}\n\n"
        "Write the Cypher query to retrieve facts needed to answer. "
        "Use case-insensitive matching and prefer CONTAINS for textual filters."
    )

    steps: list[dict[str, Any]] = []

    try:
        raw_cypher_reply, groq_gen_meta = _groq_chat(
            [
                {"role": "system", "content": gen_system},
                {"role": "user", "content": gen_user},
            ],
            model=model_name,
        )
    except Exception as e:
        logger.exception("Groq Cypher generation failed")
        err = f"weak_model_query error: {e}"
        return _finish(
            "groq_cypher_error",
            err,
            steps,
            {"error": str(e)},
        )

    steps.append(
        {
            "step": "groq_generate_cypher",
            **groq_gen_meta,
            "raw_reply": _trunc_text(raw_cypher_reply, cap),
        }
    )

    cypher = _extract_cypher_block(raw_cypher_reply)
    steps.append(
        {
            "step": "extract_cypher",
            "cypher": _trunc_text(cypher, cap) if cypher else None,
            "extracted": bool(cypher),
        }
    )
    if not cypher:
        return _finish("no_cypher_extracted", NOT_FOUND_MESSAGE, steps)

    ok, reason = _validate_read_only_cypher(cypher)
    steps.append({"step": "validate_cypher", "ok": ok, "reason": reason})
    if not ok:
        logger.info("Rejected Cypher: %s — %s", reason, cypher[:200])
        return _finish("validation_failed", NOT_FOUND_MESSAGE, steps)

    prop_allow = _neo4j_property_allowlist(database)
    lint_ok, lint_reason = _lint_cypher_property_keys(cypher, prop_allow)
    steps.append(
        {
            "step": "schema_property_lint",
            "ok": lint_ok,
            "reason": lint_reason,
            "property_catalog_size": len(prop_allow),
        }
    )
    if not lint_ok:
        logger.info("Cypher property lint failed: %s", lint_reason)
        return _finish("property_lint_failed", NOT_FOUND_MESSAGE, steps)

    driver = _neo4j_driver()
    try:
        with driver.session(database=database) as session:
            rows, err = _run_cypher_read(session, cypher)
    finally:
        driver.close()

    steps.append(
        {
            "step": "neo4j_execute",
            "cypher_executed": _trunc_text(cypher, cap),
            "row_count": len(rows),
            "error": err,
        }
    )

    if err:
        return _finish("neo4j_error", NOT_FOUND_MESSAGE, steps)

    results_json = json.dumps(rows, ensure_ascii=False, indent=2)
    if len(results_json) > 24000:
        results_json = results_json[:24000] + "\n… (truncated)"

    ans_system = (
        "You answer strictly from the JSON query results below. "
        "If they are empty or do not contain enough information, respond with exactly: "
        f"{NOT_FOUND_MESSAGE}\n"
        "Do not use outside knowledge. One or two concise sentences."
    )
    ans_user = (
        f"Question:\n{question.strip()}\n\n"
        f"Cypher used:\n```\n{cypher}\n```\n\n"
        f"Results (JSON):\n```json\n{results_json}\n```"
    )

    try:
        answer, groq_ans_meta = _groq_chat(
            [
                {"role": "system", "content": ans_system},
                {"role": "user", "content": ans_user},
            ],
            model=model_name,
        )
    except Exception as e:
        logger.exception("Groq answer synthesis failed")
        err = f"weak_model_query error: {e}"
        steps.append({"step": "groq_answer", "error": str(e)})
        return _finish("groq_answer_error", err, steps, {"error": str(e)})

    steps.append(
        {
            "step": "groq_answer",
            **groq_ans_meta,
            "answer_preview": _trunc_text(answer, cap),
        }
    )

    final = _ensure_non_empty_response(answer)
    outcome = (
        "not_found"
        if final == NOT_FOUND_MESSAGE
        else "success"
    )
    return _finish(outcome, final, steps)


def weak_model_query(question: str, user_id: str) -> str:
    return _run_in_query_thread(_cypher_student_sync, question, user_id)


def weak_model_query_with_context(question: str, user_id: str, context: Any) -> str:
    ctx = None if context is None else str(context)
    return _run_in_query_thread(_cypher_student_sync, question, user_id, ctx)


def weak_model_query_with_context_and_llm(
    question: str, user_id: str, context: Any, llm: Any
) -> str:
    ctx = None if context is None else str(context)
    return _run_in_query_thread(_cypher_student_sync, question, user_id, ctx, llm)


def init_weak_model(_user_id: str) -> None:
    """Backwards-compatible no-op; schema is fetched inside ``weak_model_query``."""
    return None
