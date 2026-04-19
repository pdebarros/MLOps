"""
Gemini Flash (Vertex AI) GraphRAG student:

- Retrieve relevant seed entities from Neo4j via fixed, read-only Cypher.
- Expand a bounded k-hop neighborhood around those seeds.
- Serialize the retrieved subgraph context.
- Ask Gemini to answer strictly from that graph context.
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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

from .config import config, google_cloud_project, project_and_location_for_vertex
from .model import (
    NOT_FOUND_MESSAGE,
    _ensure_non_empty_response,
    _neo4j_driver,
)
from .subgraph import fetch_subgraph_k_hops, serialize_subgraph

logger = logging.getLogger("eval_agent.model2")
_TRACE_PREFIX = "GEMINI_GRAPHRAG_TRACE"

_gemini_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini_graphrag")

_LLM: ChatVertexAI | None = None

# Defaults (override with env)
_GEMINI_MODEL = os.environ.get("GEMINI_GRAPHRAG_MODEL", os.environ.get("GEMINI_CYPHER_MODEL", "gemini-2.5-flash"))
_GEMINI_TEMPERATURE = float(os.environ.get("GEMINI_CYPHER_TEMPERATURE", "0.2"))
_GRAPHRAG_MAX_SEEDS = max(1, int(os.environ.get("GEMINI_GRAPHRAG_MAX_SEEDS", "8")))
_GRAPHRAG_K_HOPS = max(1, min(6, int(os.environ.get("GEMINI_GRAPHRAG_K_HOPS", "2"))))
_GRAPHRAG_MAX_STARTS = max(1, int(os.environ.get("GEMINI_GRAPHRAG_MAX_STARTS", "8")))
_GRAPHRAG_MAX_CONTEXT_CHARS = max(4000, int(os.environ.get("GEMINI_GRAPHRAG_MAX_CONTEXT_CHARS", "24000")))
_GRAPHRAG_TOP_K = max(1, int(config.GEMINI_GRAPHRAG_TOP_K))


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
    record.setdefault("trace", "gemini_graphrag_v1")
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


def _build_system_prompt(eval_agent_prompt: str | None) -> str:
    parts = [
        "You are a graph-grounded QA assistant.",
        "You receive a serialized subgraph context retrieved from Neo4j.",
        "Answer ONLY from that context.",
        f"If the context is insufficient, answer exactly: {NOT_FOUND_MESSAGE}",
        "Keep the answer concise (1-3 sentences).",
    ]
    if eval_agent_prompt:
        parts.extend(["", "## Evaluator / workflow context (from eval agent)", eval_agent_prompt.strip()])
    return "\n".join(parts)


def _question_keywords(question: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_./:-]{1,}", (question or "").lower())
    stop = {
        "what",
        "where",
        "when",
        "which",
        "who",
        "does",
        "is",
        "are",
        "the",
        "and",
        "for",
        "from",
        "with",
        "that",
        "this",
        "into",
        "about",
        "explain",
        "describe",
    }
    uniq: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if len(t) < 3 or t in stop:
            continue
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq[:20]


def _retrieve_seed_ids(database: str, question: str) -> list[str]:
    kws = _question_keywords(question)
    if not kws:
        kws = [question.strip().lower()[:64]]
    driver = _neo4j_driver()
    try:
        with driver.session(database=database) as session:
            rows = session.run(
                """
                UNWIND $kws AS kw
                MATCH (n:Entity)
                WHERE toLower(toString(n.id)) CONTAINS kw
                   OR toLower(toString(coalesce(n.kind, ''))) CONTAINS kw
                WITH n, count(*) AS hits
                ORDER BY hits DESC, size(toString(n.id)) ASC
                LIMIT $lim
                RETURN n.id AS id
                """,
                kws=kws,
                lim=_GRAPHRAG_MAX_SEEDS,
            )
            ids = [str(r["id"]) for r in rows if r.get("id")]
    finally:
        driver.close()
    return ids


def _retrieve_seed_ids_embedding_graphrag(database: str, question: str) -> tuple[list[str], str | None]:
    """
    Retrieve candidate seed ids using neo4j_graphrag vector retrieval.

    Returns (seed_ids, error_message). On any error, callers should fall back
    to keyword matching so experiments keep running.
    """
    try:
        from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
        from neo4j_graphrag.retrievers import VectorRetriever
    except Exception as e:
        return [], f"neo4j_graphrag import failed: {e}"

    driver = _neo4j_driver()
    try:
        try:
            embedder = SentenceTransformerEmbeddings(
                model=config.GEMINI_GRAPHRAG_EMBED_MODEL
            )
            retriever = VectorRetriever(
                driver=driver,
                index_name=config.GEMINI_GRAPHRAG_VECTOR_INDEX,
                embedder=embedder,
            )
            hits = retriever.search(query_text=question, top_k=_GRAPHRAG_TOP_K)
        except Exception as e:
            return [], f"vector retrieval failed: {e}"
    finally:
        driver.close()

    out: list[str] = []
    seen: set[str] = set()
    for hit in hits or []:
        # neo4j_graphrag result shapes vary by version; handle common patterns.
        rec = None
        if isinstance(hit, dict):
            rec = hit
        else:
            rec = getattr(hit, "record", None) or getattr(hit, "item", None)
        if rec is None:
            continue
        nid = None
        if isinstance(rec, dict):
            nid = rec.get("id")
            if nid is None and "node" in rec and isinstance(rec["node"], dict):
                nid = rec["node"].get("id")
        if nid is None:
            node = getattr(rec, "node", None)
            if isinstance(node, dict):
                nid = node.get("id")
        if nid is None:
            continue
        nid_s = str(nid)
        if nid_s in seen:
            continue
        seen.add(nid_s)
        out.append(nid_s)
        if len(out) >= _GRAPHRAG_MAX_SEEDS:
            break
    return out, None


def _build_graph_context(database: str, question: str) -> tuple[str, dict[str, Any]]:
    """
    Retrieve GraphRAG context via seed entity matching + k-hop expansion.
    Returns (serialized_context, retrieval_meta).
    """
    mode = (config.GEMINI_STUDENT_RETRIEVAL_MODE or "keyword").strip().lower()
    retrieval_error: str | None = None
    if mode == "embedding_graphrag":
        seeds, retrieval_error = _retrieve_seed_ids_embedding_graphrag(database, question)
        if not seeds:
            seeds = _retrieve_seed_ids(database, question)
    else:
        seeds = _retrieve_seed_ids(database, question)
    if not seeds:
        return "", {
            "seed_ids": [],
            "nodes": 0,
            "edges": 0,
            "retrieval_mode": mode,
            "retrieval_error": retrieval_error,
        }

    driver = _neo4j_driver()
    try:
        all_nodes: dict[str, dict[str, Any]] = {}
        all_edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        for sid in seeds:
            nodes, edges = fetch_subgraph_k_hops(
                driver,
                needle=None,
                exact_id=sid,
                k=_GRAPHRAG_K_HOPS,
                max_starts=_GRAPHRAG_MAX_STARTS,
                database=database,
            )
            for n in nodes:
                all_nodes[str(n.get("id"))] = n
            for e in edges:
                key = (str(e.get("src")), str(e.get("dst")), str(e.get("type") or "RELATED_TO"))
                all_edges[key] = e
    finally:
        driver.close()

    nodes_list = list(all_nodes.values())
    edges_list = list(all_edges.values())
    context = serialize_subgraph(
        nodes_list,
        edges_list,
        title=f"GraphRAG neighborhoods for question: {question[:120]}",
    )
    if len(context) > _GRAPHRAG_MAX_CONTEXT_CHARS:
        context = context[:_GRAPHRAG_MAX_CONTEXT_CHARS] + "\n... (truncated)"
    meta = {
        "seed_ids": seeds,
        "nodes": len(nodes_list),
        "edges": len(edges_list),
        "k_hops": _GRAPHRAG_K_HOPS,
        "retrieval_mode": mode,
        "retrieval_error": retrieval_error,
    }
    return context, meta


def _gemini_multi_step_sync(
    question: str,
    user_id: str,
    eval_agent_prompt: str | None = None,
) -> str:
    session_id = str(uuid.uuid4())
    database = (user_id or "").strip()
    q_preview = _trunc_trace_text(question.strip())

    def _finish(
        outcome: str,
        returned: str,
        *,
        retrieval_meta: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        rec: dict[str, Any] = {
            "session_id": session_id,
            "neo4j_database": database or None,
            "gemini_model": _GEMINI_MODEL,
            "question_preview": q_preview,
            "has_eval_agent_prompt": bool((eval_agent_prompt or "").strip()),
            "retrieval": retrieval_meta or {},
            "final_outcome": outcome,
            "returned_preview": _trunc_trace_text(returned),
        }
        if extra:
            rec.update(extra)
        _emit_gemini_session_trace(rec)
        return returned

    if not database:
        return _finish("no_database", NOT_FOUND_MESSAGE, retrieval_meta={})

    graph_context, retrieval_meta = _build_graph_context(database, question)
    if not graph_context:
        return _finish("no_graph_context", NOT_FOUND_MESSAGE, retrieval_meta=retrieval_meta)

    system = _build_system_prompt(eval_agent_prompt)
    llm = _get_chat_vertex()
    messages: list[Any] = [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Question:\n{question.strip()}\n\n"
                f"Retrieved graph context:\n{graph_context}\n\n"
                "Answer from this context only."
            )
        ),
    ]
    try:
        resp = llm.invoke(messages)
        raw = (resp.content or "").strip()
    except Exception as e:
        logger.exception("Gemini GraphRAG synthesis failed")
        err = f"gemini_flash_kg_query error: {e}"
        return _finish(
            "gemini_invoke_error",
            err,
            retrieval_meta=retrieval_meta,
            extra={"error": str(e)},
        )

    final = _ensure_non_empty_response(raw)
    outcome = "not_found" if final == NOT_FOUND_MESSAGE else "success"
    return _finish(outcome, final, retrieval_meta=retrieval_meta)


def gemini_flash_kg_query(
    question: str,
    user_id: str,
    *,
    eval_agent_prompt: str | None = None,
) -> str:
    """
    Run Gemini Flash on Vertex with a GraphRAG-style retrieval path.

    The function does NOT generate Cypher queries from the model. Instead, it uses fixed,
    internal retrieval Cypher to build a bounded graph context (seed matches + k-hop
    neighborhoods), then asks Gemini to answer strictly from that retrieved context.

    Parameters
    ----------
    question:
        Natural-language question.
    user_id:
        Neo4j logical database name.
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
