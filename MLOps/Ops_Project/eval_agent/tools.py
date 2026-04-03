"""Tools for eval_agent: Vertex RAG Engine + Neo4j KG neighborhoods."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud import storage
from neo4j import GraphDatabase

from .config import (
    Config,
    normalize_user_id,
    project_and_location_for_vertex,
    scoring_blob_path_for_user,
)
from .subgraph import fetch_subgraph_k_hops, serialize_subgraph

logger = logging.getLogger("eval_agent.tools")

_vertex_inited = False


def _neo4j_driver():
    return GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )


def _format_retrieval_response(response: Any) -> str:
    rag_ctxs = getattr(response, "contexts", None)
    if rag_ctxs is None:
        return "(No contexts field in response.)"
    inner = getattr(rag_ctxs, "contexts", None) or []
    if not inner:
        return "(RAG returned no chunks.)"
    parts: list[str] = []
    for i, ctx in enumerate(inner, 1):
        text = getattr(ctx, "text", "") or ""
        src = getattr(ctx, "source_uri", "") or getattr(ctx, "source_display_name", "") or ""
        score = getattr(ctx, "score", None)
        score_s = f"{score:.4f}" if score is not None else "n/a"
        parts.append(
            f"--- Chunk {i} (score={score_s}) ---\n"
            f"Source: {src}\n\n{text.strip()}\n"
        )
    return "\n".join(parts)


def _ensure_vertex_init(corpus_resource: str) -> None:
    global _vertex_inited
    if _vertex_inited:
        return
    import vertexai

    project, location = project_and_location_for_vertex(corpus_resource)
    vertexai.init(project=project, location=location)
    _vertex_inited = True
    logger.info("vertexai.init(project=%s, location=%s)", project, location)


def retrieve_from_rag_corpus(query: str) -> str:
    """
    Retrieve relevant text chunks from the configured Vertex AI RAG Engine corpus.

    Use this to ground evaluation: the corpus should contain the same summaries / source
    material the KG was built from. Query with file paths, module names, class names,
    or concepts you see in the subgraph.

    Args:
        query: Natural language or keyword query for semantic search over the corpus.

    Returns:
        Concatenated retrieved chunks with source URIs, or an error message string.
    """
    corpus = Config.VERTEX_RAG_CORPUS
    if not corpus:
        return (
            "RAG is not configured. Set VERTEX_RAG_CORPUS to the full corpus resource name "
            "(projects/.../locations/.../ragCorpora/...) or set RAG_CORPUS_ID with "
            "GOOGLE_CLOUD_PROJECT and VERTEX_LOCATION."
        )
    q = (query or "").strip()
    if not q:
        return "Error: empty query."

    try:
        _ensure_vertex_init(corpus)
        from vertexai import rag

        rag_resource = rag.RagResource(rag_corpus=corpus)
        cfg = rag.RagRetrievalConfig(top_k=Config.RAG_TOP_K)
        if Config.RAG_VECTOR_DISTANCE_THRESHOLD is not None:
            cfg.filter = rag.Filter(
                vector_distance_threshold=Config.RAG_VECTOR_DISTANCE_THRESHOLD
            )

        resp = rag.retrieval_query(
            text=q,
            rag_resources=[rag_resource],
            rag_retrieval_config=cfg,
        )
        return _format_retrieval_response(resp)
    except Exception as e:
        logger.exception("RAG retrieval failed")
        return f"RAG retrieval error: {e}"


def list_entity_id_samples(limit: int = 15) -> str:
    """
    List a random sample of Entity node ids from Neo4j to choose neighborhood seeds.

    Use before `fetch_neighborhood_from_neo4j` when you need diverse starting points
    without scanning the whole graph. Does not traverse edges.

    Args:
        limit: How many entity ids to return (capped by config).

    Returns:
        Numbered list of entity ids, or an error message.
    """
    cap = max(1, min(int(limit), Config.EVAL_ENTITY_LIST_LIMIT))
    try:
        driver = _neo4j_driver()
        try:
            with driver.session() as session:
                rows = session.run(
                    """
                    MATCH (n:Entity)
                    WITH n ORDER BY rand()
                    LIMIT $lim
                    RETURN n.id AS id
                    """,
                    lim=cap,
                )
                ids = [str(r["id"]) for r in rows if r.get("id")]
        finally:
            driver.close()
    except Exception as e:
        logger.exception("Neo4j list_entity_id_samples failed")
        return f"Neo4j error listing entities: {e}"

    if not ids:
        return "No :Entity nodes found in Neo4j (empty graph or wrong database)."
    lines = [f"{i + 1}. `{id_}`" for i, id_ in enumerate(ids)]
    return "Sample Entity ids (random):\n" + "\n".join(lines)


def fetch_neighborhood_from_neo4j(
    exact_entity_id: str | None = None,
    id_substring: str | None = None,
    k_hops: int | None = None,
    max_start_nodes: int | None = None,
) -> str:
    """
    Load a k-hop neighborhood around one or more :Entity nodes (same pattern as
    KG_agent/kg_reconstruct_test.py). Use this to inspect the local structure of the KG.

    Provide either `exact_entity_id` (single node id) or `id_substring` (matches
    Entity.id CONTAINS). When using substring, up to `max_start_nodes` start nodes
    are used.

    Args:
        exact_entity_id: Exact Entity.id (preferred for a known seed from the list tool).
        id_substring: Substring match against Entity.id (e.g. a filename fragment).
        k_hops: Graph hop depth 1–10 (default from EVAL_DEFAULT_K).
        max_start_nodes: Max start entities when matching by substring (default EVAL_MAX_STARTS).

    Returns:
        Markdown summary of nodes and typed :REL edges in the neighborhood.
    """
    exact = (exact_entity_id or "").strip() or None
    needle = (id_substring or "").strip() or None
    if exact is None and not needle:
        return "Error: provide `exact_entity_id` or `id_substring`."

    k = int(k_hops) if k_hops is not None else Config.EVAL_DEFAULT_K
    ms = int(max_start_nodes) if max_start_nodes is not None else Config.EVAL_MAX_STARTS

    try:
        driver = _neo4j_driver()
        try:
            nodes, edges = fetch_subgraph_k_hops(
                driver,
                needle=needle,
                exact_id=exact,
                k=k,
                max_starts=ms,
            )
        finally:
            driver.close()
    except Exception as e:
        logger.exception("Neo4j fetch_neighborhood_from_neo4j failed")
        return f"Neo4j error fetching neighborhood: {e}"

    title = exact or needle or "unknown"
    return serialize_subgraph(nodes, edges, title=f"{title} (k={k})")


def _clamp_unit(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _parse_per_sample_scores_json(raw: str | None) -> tuple[list[dict[str, Any]] | None, str | None]:
    """
    Parse and normalize per-sample score rows. Returns (list, error_message).
    Each element must be a dict with criterion_1_score and criterion_2_score (0..1).
    """
    if raw is None or not str(raw).strip():
        return [], None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return None, f"per_sample_scores_json is not valid JSON: {e}"
    if not isinstance(data, list):
        return None, "per_sample_scores_json must be a JSON array."
    out: list[dict[str, Any]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return None, f"per_sample_scores_json[{i}] must be an object."
        c1 = item.get("criterion_1_score", item.get("c1"))
        c2 = item.get("criterion_2_score", item.get("c2"))
        if c1 is None or c2 is None:
            return (
                None,
                f"per_sample_scores_json[{i}] must include criterion_1_score and criterion_2_score.",
            )
        row: dict[str, Any] = dict(item)
        row["criterion_1_score"] = round(_clamp_unit(float(c1)), 4)
        row["criterion_2_score"] = round(_clamp_unit(float(c2)), 4)
        row.pop("c1", None)
        row.pop("c2", None)
        out.append(row)
    return out, None


def save_scores_to_gcs(
    user_id: str,
    criterion_1_score: float,
    criterion_2_score: float,
    samples_evaluated: int,
    per_sample_scores_json: str | None = None,
    overall_score: float | None = None,
    aggregation_rule: str | None = None,
) -> str:
    """
    Persist **final aggregated** evaluation scores to Google Cloud Storage.

    Call **once** at the end of a run, after you computed **averages** of criterion 1 and 2
    across the neighborhoods you evaluated. Appends one JSON line under the user's folder:
    ``gs://<GCS_BUCKET_NAME>/<user_id>/scoring`` (NDJSON). Creates the object if missing.

    The user must have provided **user_id** at the start (same folder convention as
    ``pipeline2.py`` for ``codebases-04-26``).

    Args:
        user_id: GCS user prefix, e.g. ``u_123`` (must match the session’s user).
        criterion_1_score: Average score on [0.0, 1.0] for RAG / structural alignment.
        criterion_2_score: Average score on [0.0, 1.0] for programming-knowledge capture.
        samples_evaluated: Number of neighborhoods (or samples) those averages are based on.
        per_sample_scores_json: JSON array of per-neighborhood scores, same order as evaluated.
            Each object must include ``criterion_1_score`` and ``criterion_2_score`` (0.0–1.0).
            Recommended keys: ``sample_index``, ``seed_entity_id`` (or ``label``), optional ``notes``.
            Example: ``[{"sample_index":1,"seed_entity_id":"pkg/a.py","criterion_1_score":0.7,"criterion_2_score":0.65}]``
        overall_score: Optional combined score (e.g. mean of the two criteria).
        aggregation_rule: One short line on how averages were computed (optional).

    Returns:
        Confirmation with `gs://` URI, or an error message.
    """
    bucket_name = (Config.GCS_BUCKET_NAME or "").strip()
    if not bucket_name:
        return "GCS is not configured: set GCS_BUCKET_NAME."

    try:
        uid = normalize_user_id(user_id)
    except ValueError as e:
        return f"Invalid user_id: {e}"

    per_sample, ps_err = _parse_per_sample_scores_json(per_sample_scores_json)
    if ps_err:
        return ps_err

    c1 = _clamp_unit(criterion_1_score)
    c2 = _clamp_unit(criterion_2_score)
    ov = None if overall_score is None else _clamp_unit(overall_score)
    n = max(0, int(samples_evaluated))
    rule = (aggregation_rule or "").strip() or None

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": uid,
        "criterion_1_score": round(c1, 4),
        "criterion_2_score": round(c2, 4),
        "overall_score": round(ov, 4) if ov is not None else None,
        "samples_evaluated": n,
        "aggregation_rule": rule,
        "per_sample_scores": per_sample,
    }
    line = json.dumps(record, ensure_ascii=False) + "\n"
    object_name = scoring_blob_path_for_user(uid)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        if blob.exists():
            existing = blob.download_as_text(encoding="utf-8")
            body = existing + line
        else:
            body = line
        blob.upload_from_string(
            body,
            content_type="application/x-ndjson; charset=utf-8",
        )
    except Exception as e:
        logger.exception("save_scores_to_gcs failed")
        return f"GCS error: {e}"

    return (
        f"Appended scoring record to gs://{bucket_name}/{object_name} "
        f"(criterion_1={record['criterion_1_score']}, criterion_2={record['criterion_2_score']}, "
        f"samples={n})."
    )
