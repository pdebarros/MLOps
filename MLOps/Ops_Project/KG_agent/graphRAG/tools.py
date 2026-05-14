"""
Tools for the graphRAG agent.

Implements a **two-source hybrid GraphRAG strategy** combining the Neo4j
experience knowledge graph with a Vertex AI RAG Engine corpus of GCS-stored
chunks. Both sources are tenant-scoped to the same normalised user id.

Workflow:

  Step 0 — assess_query_difficulty:
      Heuristically grade the query as 'easy', 'moderate', or 'hard' to choose
      ``top_k`` (#seeds) and ``hop_depth`` (neighborhood radius).

  Step 1a — vector_seed_search (graph source):
      Embed the query, ANN-query Neo4j vector index ``exp_entity_embedding``
      (scoped to ``:User_<uid>``).

  Step 1b — rag_corpus_query (RAG source):
      Query the user's Vertex AI RAG Engine corpus (per-tenant corpus or
      shared corpus with metadata filter). Uses ADC for now; will be a
      service account in Agent Engine.

  Step 2 — fetch_neighborhood (graph source):
      For each seed node, expand a k-hop neighborhood. Cached in session state.

  Step 3a — score_neighborhood:
      Rate each neighborhood on [0.0, 1.0]; stored in
      ``state['seed_scores'][seed_id]``.

  Step 3b — score_rag_evidence:
      Rate the RAG corpus retrieval as a whole on [0.0, 1.0]; stored in
      ``state['rag_score']``.

  Step 4 — set_source_weights:
      Agent decides how to weight graph vs. RAG when synthesizing. Weights
      stored in ``state['source_weights']`` and must sum to 1.0.

  Step 5 — synthesize_evidence:
      Builds a final consolidated evidence block annotated with the source
      weights, ready for grounded answer generation.

All tools accept the ``tenant_id`` (= normalised user_id) so multi-tenancy is
enforced on every Cypher call via the ``:User_<uid>`` label and ``tenantId``
property, and on every RAG query via per-tenant corpus or metadata filter.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any

from google.adk.tools import ToolContext
from neo4j import GraphDatabase

from .config import Config

logger = logging.getLogger("graphRAG.tools")

# Lazy singletons (built once per process)
_neo4j_driver = None
_embedding_model = None
_vertex_inited = False


# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        if not Config.NEO4J_URI or not Config.NEO4J_USER or not Config.NEO4J_PASSWORD:
            raise RuntimeError(
                "Neo4j is not configured. Set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD."
            )
        _neo4j_driver = GraphDatabase.driver(
            Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
    return _neo4j_driver


def _session_kwargs() -> dict[str, Any]:
    db = (Config.NEO4J_DATABASE or "").strip()
    return {"database": db} if db else {}


def _sanitize_tenant_label(uid: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", uid)
    if safe and safe[0].isdigit():
        safe = "U" + safe
    return f"User_{safe}"


def _normalize_user_id(uid: str) -> str:
    """Match experience_pipeline.normalize_user_id semantics."""
    return (uid or "").strip().lower().replace(" ", "_")


@functools.lru_cache(maxsize=512)
def _bq_rag_corpus_id_for_user(normalized_user_id: str, table_fqn: str) -> str | None:
    """
    Read ``rag_corpus_id`` from the gitai users table (same contract as gitai-upload-api).

    ``table_fqn`` must be ``project.dataset.table``. Cached per (user, table).
    """
    parts = table_fqn.split(".")
    if len(parts) != 3:
        logger.error("BigQuery users table FQN must be project.dataset.table, got: %s", table_fqn)
        return None
    bq_project = parts[0]
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise RuntimeError(
            "google-cloud-bigquery is required for BigQuery RAG corpus resolution. "
            "Add it to your environment (see graphRAG/requirements.txt)."
        ) from e

    client = bigquery.Client(project=bq_project)
    sql = f"""
        SELECT rag_corpus_id
        FROM `{table_fqn}`
        WHERE user_id = @user_id
        LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", normalized_user_id),
        ]
    )
    try:
        rows = list(client.query(sql, job_config=cfg).result())
    except Exception:
        logger.exception(
            "BigQuery rag_corpus_id lookup failed (user_id=%s, table=%s)",
            normalized_user_id,
            table_fqn,
        )
        raise
    if not rows:
        return None
    row = dict(rows[0])
    raw = row.get("rag_corpus_id")
    if raw is None:
        return None
    s = str(raw).strip()
    return s or None


def _get_embedder():
    global _embedding_model
    if _embedding_model is None:
        from langchain_google_vertexai import VertexAIEmbeddings  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "model_name": Config.EMBEDDING_MODEL,
            "location": Config.VERTEX_LOCATION,
        }
        if Config.GOOGLE_CLOUD_PROJECT:
            kwargs["project"] = Config.GOOGLE_CLOUD_PROJECT
        _embedding_model = VertexAIEmbeddings(**kwargs)
        logger.info(
            "Embedder ready: %s @ %s", Config.EMBEDDING_MODEL, Config.VERTEX_LOCATION
        )
    return _embedding_model


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


# ── Tool 1: query difficulty assessment ──────────────────────────────────────
def assess_query_difficulty(query: str, tool_context: ToolContext) -> str:
    """
    Heuristically classify the user's query as 'easy', 'moderate', or 'hard'
    and pick retrieval hyperparameters (top_k seeds and hop_depth).

    Heuristics:
      easy     — short, single-concept lookup ("what is X?"). Few seeds, 1 hop.
      moderate — relational / comparative ("how does X relate to Y?"). More
                 seeds, 2 hops.
      hard     — multi-concept reasoning, broad scope ("how experienced is the
                 user with async patterns?"). Many seeds, 3 hops.

    Decisions and the chosen hyperparameters are written to session state under
    keys ``query_difficulty``, ``top_k``, ``hop_depth`` so downstream tools
    (and the agent itself) can read them.

    Args:
        query: The user's natural-language question.

    Returns:
        A short markdown summary of the chosen difficulty + parameters.
    """
    q = (query or "").strip()
    if not q:
        return "Error: empty query."

    # Cheap heuristic signals — avoids an extra LLM round-trip.
    word_count = len(q.split())
    has_multi_concept = any(
        marker in q.lower()
        for marker in (
            " and ", " or ", " vs ", "compare", "relate", "between",
            "combine", "together", "overall", "across",
        )
    )
    has_reasoning = any(
        marker in q.lower()
        for marker in ("why", "how does", "explain", "trace", "walk", "chain")
    )
    has_breadth = any(
        marker in q.lower()
        for marker in ("all", "every", "summary", "overview", "experience")
    )

    score = 0
    if word_count > 12:
        score += 1
    if word_count > 22:
        score += 1
    if has_multi_concept:
        score += 1
    if has_reasoning:
        score += 1
    if has_breadth:
        score += 1

    if score <= 1:
        difficulty = "easy"
        top_k = _clamp(Config.MIN_TOP_K, Config.MIN_TOP_K, Config.MAX_TOP_K)
        hop_depth = _clamp(Config.MIN_HOPS, Config.MIN_HOPS, Config.MAX_HOPS)
    elif score <= 3:
        difficulty = "moderate"
        top_k = _clamp((Config.MIN_TOP_K + Config.MAX_TOP_K) // 2,
                       Config.MIN_TOP_K, Config.MAX_TOP_K)
        hop_depth = _clamp(2, Config.MIN_HOPS, Config.MAX_HOPS)
    else:
        difficulty = "hard"
        top_k = _clamp(Config.MAX_TOP_K, Config.MIN_TOP_K, Config.MAX_TOP_K)
        hop_depth = _clamp(Config.MAX_HOPS, Config.MIN_HOPS, Config.MAX_HOPS)

    tool_context.state["query"] = q
    tool_context.state["query_difficulty"] = difficulty
    tool_context.state["top_k"] = top_k
    tool_context.state["hop_depth"] = hop_depth
    tool_context.state.setdefault("seed_scores", {})

    logger.info(
        "Difficulty: %s (score=%d) → top_k=%d, hop_depth=%d",
        difficulty, score, top_k, hop_depth,
    )

    return (
        f"## Query difficulty assessment\n"
        f"- difficulty: **{difficulty}** (signal score {score})\n"
        f"- top_k seeds: **{top_k}**\n"
        f"- hop depth: **{hop_depth}**\n"
        f"- bounds: top_k ∈ [{Config.MIN_TOP_K}, {Config.MAX_TOP_K}], "
        f"hops ∈ [{Config.MIN_HOPS}, {Config.MAX_HOPS}]\n\n"
        "These were saved to session state. You may override them by passing "
        "explicit ``top_k``/``hop_depth`` values to ``vector_seed_search`` and "
        "``fetch_neighborhood``."
    )


# ── Tool 2: vector ANN seed search ───────────────────────────────────────────
def vector_seed_search(
    query: str,
    tenant_id: str,
    tool_context: ToolContext,
    top_k: int | None = None,
) -> str:
    """
    Embed the query and run an approximate-nearest-neighbour search against the
    Neo4j vector index ``exp_entity_embedding``, scoped to the given tenant.

    The returned seed entity ids are stored in session state under
    ``seed_entity_ids`` so subsequent tools can iterate them.

    Args:
        query: User question to embed.
        tenant_id: Raw user id; will be normalised. Used to filter on
                   ``tenantId`` property so cross-tenant results are excluded.
        top_k: Override the auto-chosen top_k (clamped to config bounds).

    Returns:
        Markdown table of the top-k seeds with their kind, score, and text.
    """
    uid = _normalize_user_id(tenant_id)
    if not uid:
        return "Error: tenant_id is required."

    if top_k is None:
        top_k = int(tool_context.state.get("top_k", Config.MIN_TOP_K))
    top_k = _clamp(int(top_k), Config.MIN_TOP_K, Config.MAX_TOP_K)

    user_label = _sanitize_tenant_label(uid)

    try:
        embedder = _get_embedder()
        vec = embedder.embed_query(query.strip())
    except Exception as exc:
        logger.exception("Embedding failed")
        return f"Embedding error: {exc}"

    try:
        driver = _get_driver()
        with driver.session(**_session_kwargs()) as session:
            # We over-fetch by 4× then filter to tenant — vector index has no
            # native filter clause until Neo4j 5.18+, so this is the portable
            # approach that works with AuraDB Free.
            rows = session.run(
                f"""
                CALL db.index.vector.queryNodes($index_name, $fetch_n, $vec)
                YIELD node, score
                WHERE node:`{user_label}` AND node.tenantId = $tid
                RETURN node.id   AS id,
                       node.kind AS kind,
                       node.text AS text,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """,
                index_name=Config.VECTOR_INDEX_NAME,
                fetch_n=max(top_k * 4, 20),
                vec=vec,
                tid=uid,
                top_k=top_k,
            )
            seeds = [
                {
                    "id": r["id"],
                    "kind": r["kind"] or "Entity",
                    "text": (r["text"] or "")[:300],
                    "score": float(r["score"]),
                }
                for r in rows
            ]
    except Exception as exc:
        logger.exception("Vector search failed")
        return f"Neo4j vector search error: {exc}"

    if not seeds:
        return (
            "No seeds found. Possible causes: vector index missing, no nodes "
            "embedded for this tenant yet, or the user_id is wrong."
        )

    tool_context.state["tenant_id"] = uid
    tool_context.state["seed_entity_ids"] = [s["id"] for s in seeds]
    tool_context.state["seed_metadata"] = seeds

    lines = ["## Vector seed search", f"tenant: `{uid}` | top_k: {top_k}", ""]
    lines.append("| # | id | kind | similarity | preview |")
    lines.append("|---|----|------|------------|---------|")
    for i, s in enumerate(seeds, 1):
        preview = s["text"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | `{s['id']}` | {s['kind']} | {s['score']:.3f} | {preview} |"
        )
    return "\n".join(lines)


# ── Tool 3: neighborhood expansion ───────────────────────────────────────────
def fetch_neighborhood(
    seed_entity_id: str,
    tenant_id: str,
    tool_context: ToolContext,
    hop_depth: int | None = None,
) -> str:
    """
    Expand a k-hop neighborhood around a single seed node (tenant-scoped) and
    return a serialized view of nodes + relationships.

    The full subgraph is also saved to ``tool_context.state['neighborhoods'][seed_id]``
    so other tools (notably ``score_neighborhood``) can re-read it without
    re-querying Neo4j.

    Args:
        seed_entity_id: Exact Entity.id of one node returned by
                        ``vector_seed_search``.
        tenant_id: Raw user id; will be normalised.
        hop_depth: Override the auto-chosen hop_depth (clamped to config bounds).

    Returns:
        Markdown with a Nodes section and an Edges section.
    """
    uid = _normalize_user_id(tenant_id)
    if not uid or not seed_entity_id:
        return "Error: tenant_id and seed_entity_id are required."

    if hop_depth is None:
        hop_depth = int(tool_context.state.get("hop_depth", Config.MIN_HOPS))
    hop_depth = _clamp(int(hop_depth), Config.MIN_HOPS, Config.MAX_HOPS)

    user_label = _sanitize_tenant_label(uid)

    try:
        driver = _get_driver()
        with driver.session(**_session_kwargs()) as session:
            # Variable-length pattern bounded by hop_depth. We materialize ALL
            # paths of length <= k from the seed in either direction, then
            # de-duplicate nodes and edges in Python.
            res = session.run(
                f"""
                MATCH (seed:Entity:`{user_label}` {{id: $sid, tenantId: $tid}})
                CALL {{
                    WITH seed
                    MATCH p = (seed)-[*1..{hop_depth}]-(other:Entity:`{user_label}`)
                    WHERE other.tenantId = $tid
                    RETURN nodes(p) AS ns, relationships(p) AS rs
                    LIMIT $edge_cap
                }}
                RETURN ns, rs
                """,
                sid=seed_entity_id,
                tid=uid,
                edge_cap=Config.MAX_NEIGHBOR_EDGES,
            )

            seen_nodes: dict[str, dict[str, Any]] = {}
            seen_edges: dict[tuple[str, str, str], dict[str, Any]] = {}

            for row in res:
                for n in row["ns"]:
                    nid = n.get("id")
                    if not nid or nid in seen_nodes:
                        continue
                    seen_nodes[nid] = {
                        "id": nid,
                        "kind": n.get("kind") or "Entity",
                        "text": (n.get("text") or "")[:300],
                    }
                for r in row["rs"]:
                    src = r.start_node.get("id")
                    dst = r.end_node.get("id")
                    rtype = r.get("type") or r.type
                    if not src or not dst or not rtype:
                        continue
                    key = (src, rtype, dst)
                    if key in seen_edges:
                        continue
                    rel_props = {
                        k: v for k, v in dict(r).items()
                        if k not in ("type", "tenantId")
                    }
                    seen_edges[key] = {
                        "source": src,
                        "type": rtype,
                        "target": dst,
                        "properties": rel_props,
                    }
                if len(seen_nodes) >= Config.MAX_NEIGHBOR_NODES:
                    break
    except Exception as exc:
        logger.exception("Neighborhood fetch failed")
        return f"Neo4j neighborhood error: {exc}"

    if seed_entity_id not in seen_nodes:
        # Always include the seed itself, even if it has no edges within depth.
        try:
            with _get_driver().session(**_session_kwargs()) as session:
                row = session.run(
                    f"""
                    MATCH (n:Entity:`{user_label}` {{id: $sid, tenantId: $tid}})
                    RETURN n.id AS id, n.kind AS kind, n.text AS text
                    """,
                    sid=seed_entity_id, tid=uid,
                ).single()
                if row:
                    seen_nodes[seed_entity_id] = {
                        "id": row["id"], "kind": row["kind"] or "Entity",
                        "text": (row["text"] or "")[:300],
                    }
        except Exception:
            pass

    nodes = list(seen_nodes.values())
    edges = list(seen_edges.values())

    # Cache for later tools
    nbhd_state = tool_context.state.setdefault("neighborhoods", {})
    nbhd_state[seed_entity_id] = {
        "seed": seed_entity_id,
        "hop_depth": hop_depth,
        "nodes": nodes,
        "edges": edges,
    }

    if not nodes and not edges:
        return f"Neighborhood for `{seed_entity_id}` (k={hop_depth}) — empty."

    lines = [
        f"## Neighborhood — seed `{seed_entity_id}` (k={hop_depth})",
        f"nodes: {len(nodes)} | edges: {len(edges)}",
        "",
        "### Nodes",
    ]
    for n in nodes[:40]:
        text = n["text"].replace("\n", " ")
        lines.append(f"- `{n['id']}` ({n['kind']}) — {text}")

    lines.append("")
    lines.append("### Edges")
    for e in edges[:80]:
        props = ", ".join(f"{k}={v}" for k, v in e["properties"].items() if v)
        suffix = f"  [{props}]" if props else ""
        lines.append(f"- `{e['source']}` —[{e['type']}]→ `{e['target']}`{suffix}")

    return "\n".join(lines)


# ── Tool 4: score a neighborhood for relevance ───────────────────────────────
def score_neighborhood(
    seed_entity_id: str,
    confidence: float,
    rationale: str,
    tool_context: ToolContext,
) -> str:
    """
    Persist a confidence score in [0.0, 1.0] for how well a neighborhood
    addresses the user's query.

    The agent calls this once per neighborhood it has inspected (via
    ``fetch_neighborhood``). Scores are stored in
    ``tool_context.state['seed_scores'][seed_entity_id]`` and aggregated by
    ``assemble_context``.

    Args:
        seed_entity_id: The seed whose neighborhood is being rated.
        confidence: Float in [0.0, 1.0]. Closer to 1.0 means the neighborhood
                    strongly addresses the user's prompt.
        rationale: One- or two-sentence justification (kept for transparency).

    Returns:
        Confirmation string with the stored score.
    """
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "Error: confidence must be a float in [0.0, 1.0]."
    c = max(0.0, min(1.0, c))

    seed = (seed_entity_id or "").strip()
    if not seed:
        return "Error: seed_entity_id is required."

    seed_scores = tool_context.state.setdefault("seed_scores", {})
    seed_scores[seed] = {
        "confidence": c,
        "rationale": (rationale or "").strip(),
    }
    logger.info("Scored neighborhood %s → %.3f", seed, c)
    return f"Stored confidence={c:.3f} for seed `{seed}`."


# ── Tool 5: assemble final retrieval context ─────────────────────────────────
def assemble_context(
    tool_context: ToolContext,
    min_confidence: float = 0.4,
    max_neighborhoods: int = 5,
) -> str:
    """
    Build the final evidence block from the highest-scoring neighborhoods.

    Reads the neighborhoods cached during ``fetch_neighborhood`` and the
    confidence scores written by ``score_neighborhood``, then returns a
    Markdown block sorted by confidence (descending), limited to
    ``max_neighborhoods``, and filtered by ``min_confidence``.

    Args:
        min_confidence: Drop any neighborhood scored below this threshold.
        max_neighborhoods: Cap on how many neighborhoods to include.

    Returns:
        A consolidated context block the LLM can ground its final answer in.
    """
    nbhds: dict[str, dict[str, Any]] = tool_context.state.get("neighborhoods") or {}
    scores: dict[str, dict[str, Any]] = tool_context.state.get("seed_scores") or {}

    if not nbhds:
        return "No neighborhoods have been fetched yet."

    ranked: list[tuple[str, float, str]] = []
    for seed in nbhds:
        sc = scores.get(seed, {})
        conf = float(sc.get("confidence", 0.0))
        rationale = sc.get("rationale", "")
        if conf < float(min_confidence):
            continue
        ranked.append((seed, conf, rationale))
    ranked.sort(key=lambda x: -x[1])
    ranked = ranked[: int(max_neighborhoods)]

    if not ranked:
        return (
            f"No neighborhoods met min_confidence={min_confidence}. "
            "Lower the threshold or fetch more neighborhoods."
        )

    out = [
        "# Assembled GraphRAG context",
        f"query: {tool_context.state.get('query', '(unknown)')}",
        f"difficulty: {tool_context.state.get('query_difficulty', '(unknown)')}",
        f"included {len(ranked)} of {len(nbhds)} neighborhood(s) "
        f"(min_confidence={min_confidence})",
        "",
    ]

    for seed, conf, rationale in ranked:
        nh = nbhds[seed]
        out.append(f"## Seed `{seed}` — confidence {conf:.2f}")
        if rationale:
            out.append(f"_rationale_: {rationale}")
        out.append(f"hop_depth={nh.get('hop_depth')} | "
                   f"nodes={len(nh.get('nodes') or [])} | "
                   f"edges={len(nh.get('edges') or [])}")
        out.append("")
        out.append("### Nodes")
        for n in (nh.get("nodes") or [])[:25]:
            text = (n.get("text") or "").replace("\n", " ")
            out.append(f"- `{n['id']}` ({n.get('kind', 'Entity')}) — {text}")
        out.append("")
        out.append("### Edges")
        for e in (nh.get("edges") or [])[:50]:
            out.append(f"- `{e['source']}` —[{e['type']}]→ `{e['target']}`")
        out.append("")

    out.append("## Score summary (all neighborhoods)")
    out.append("```json")
    out.append(json.dumps(scores, indent=2, ensure_ascii=False))
    out.append("```")

    return "\n".join(out)


# ── RAG Engine corpus helpers (per-tenant, ADC) ──────────────────────────────
def _resolve_tenant_corpus(tenant_id: str) -> tuple[str, str]:
    """
    Return ``(corpus_resource, mode)`` for the tenant.

    ``mode`` is ``"from_bq"`` (corpus id loaded from BigQuery), ``"per_tenant"``
    (template path), or ``"shared"`` (single corpus + metadata filter).

    Raises ``RuntimeError`` if RAG is not configured.
    """
    table_fqn = Config.bq_users_table_fqn()
    if table_fqn:
        cid = _bq_rag_corpus_id_for_user(tenant_id, table_fqn)
        if not cid:
            raise RuntimeError(
                f"No BigQuery row or empty rag_corpus_id for user_id={tenant_id!r} "
                f"in `{table_fqn}`. Ensure the user registered and corpus provisioning completed."
            )
        if cid.startswith("projects/"):
            return cid.strip(), "from_bq"
        proj = (Config.GOOGLE_CLOUD_PROJECT or "").strip()
        loc = (Config.VERTEX_LOCATION or "us-central1").strip()
        if not proj:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT must be set when building the Vertex RAG corpus path "
                "from BigQuery rag_corpus_id (numeric id)."
            )
        corpus = f"projects/{proj}/locations/{loc}/ragCorpora/{cid.strip()}"
        return corpus, "from_bq"

    tpl = (Config.VERTEX_RAG_CORPUS_TEMPLATE or "").strip()
    if tpl:
        try:
            corpus = tpl.format(tenant=tenant_id)
        except KeyError as e:
            raise RuntimeError(
                f"VERTEX_RAG_CORPUS_TEMPLATE is missing placeholder {{tenant}}: {tpl}"
            ) from e
        return corpus, "per_tenant"

    shared = (Config.VERTEX_RAG_CORPUS or "").strip()
    if shared:
        return shared, "shared"

    raise RuntimeError(
        "RAG is not configured. Set GRAPHRAG_BQ_USERS_TABLE_REF (or "
        "GRAPHRAG_RESOLVE_RAG_FROM_BQ=1 with BQ_PROJECT_ID / GOOGLE_CLOUD_PROJECT and "
        "BQ_DATASET / BQ_USERS_TABLE), VERTEX_RAG_CORPUS_TEMPLATE, or VERTEX_RAG_CORPUS."
    )


def _ensure_vertex_init(corpus_resource: str) -> None:
    """Initialise vertexai once (idempotent). Uses ADC for now."""
    global _vertex_inited
    if _vertex_inited:
        return
    import vertexai  # noqa: PLC0415
    import re as _re  # noqa: PLC0415

    m = _re.match(
        r"^projects/(?P<proj>[^/]+)/locations/(?P<loc>[^/]+)/ragCorpora/.*$",
        corpus_resource.strip(),
    )
    if m:
        project = m.group("proj")
        location = m.group("loc")
    else:
        project = Config.GOOGLE_CLOUD_PROJECT
        location = Config.VERTEX_LOCATION
        if not project:
            raise RuntimeError(
                "Cannot init vertexai: corpus is not a full resource name and "
                "GOOGLE_CLOUD_PROJECT is not set."
            )

    vertexai.init(project=project, location=location)
    _vertex_inited = True
    logger.info("vertexai.init(project=%s, location=%s)", project, location)


def _format_rag_response(response: Any) -> tuple[str, list[dict[str, Any]]]:
    """Return (markdown_text, structured_chunks)."""
    rag_ctxs = getattr(response, "contexts", None)
    if rag_ctxs is None:
        return "(No contexts field in RAG response.)", []
    inner = getattr(rag_ctxs, "contexts", None) or []
    if not inner:
        return "(RAG returned no chunks.)", []

    md_parts: list[str] = []
    chunks: list[dict[str, Any]] = []
    for i, ctx in enumerate(inner, 1):
        text = (getattr(ctx, "text", "") or "").strip()
        src = (
            getattr(ctx, "source_uri", "")
            or getattr(ctx, "source_display_name", "")
            or ""
        )
        score = getattr(ctx, "score", None)
        score_s = f"{score:.4f}" if score is not None else "n/a"
        md_parts.append(
            f"--- Chunk {i} (score={score_s}) ---\nSource: {src}\n\n{text}\n"
        )
        chunks.append(
            {
                "index": i,
                "source": src,
                "score": float(score) if score is not None else None,
                "text": text[:1000],
            }
        )
    return "\n".join(md_parts), chunks


# ── Tool 6: RAG corpus retrieval ─────────────────────────────────────────────
def rag_corpus_query(
    query: str,
    tenant_id: str,
    tool_context: ToolContext,
    top_k: int | None = None,
) -> str:
    """
    Retrieve relevant chunks from the user's Vertex AI RAG Engine corpus.

    Uses tenant isolation in one of three modes (auto-selected by config):
      * **BigQuery**: ``rag_corpus_id`` for ``user_id = tenant_id`` in the users
        table (see ``GRAPHRAG_BQ_USERS_TABLE_REF`` / ``GRAPHRAG_RESOLVE_RAG_FROM_BQ``).
      * **Per-tenant corpus template**: each tenant has their own corpus path via
        ``VERTEX_RAG_CORPUS_TEMPLATE`` (e.g. ``...ragCorpora/gitai-{tenant}``).
      * **Shared corpus**: a single corpus via ``VERTEX_RAG_CORPUS``, with a
        metadata filter on ``VERTEX_RAG_TENANT_METADATA_KEY`` at query time.

    Authentication: Application Default Credentials for now (gcloud auth
    application-default login). When deployed to Agent Engine the same code
    will use the attached service account automatically — no changes needed.

    Args:
        query: Natural language query for semantic retrieval.
        tenant_id: Raw user id; will be normalised.
        top_k: Override RAG_TOP_K (default 8).

    Returns:
        Markdown listing the retrieved chunks with source URIs and scores.
        Structured chunks are also stored in ``state['rag_chunks']`` for
        downstream tools.
    """
    uid = _normalize_user_id(tenant_id)
    if not uid:
        return "Error: tenant_id is required."

    q = (query or "").strip()
    if not q:
        return "Error: empty query."

    if not Config.is_rag_configured():
        return (
            "RAG is not configured. Set GRAPHRAG_BQ_USERS_TABLE_REF (or "
            "GRAPHRAG_RESOLVE_RAG_FROM_BQ=1 with BQ_*), VERTEX_RAG_CORPUS_TEMPLATE, "
            "or VERTEX_RAG_CORPUS in your environment."
        )

    k = int(top_k) if top_k is not None else Config.RAG_TOP_K
    k = max(1, min(k, 50))

    try:
        corpus, mode = _resolve_tenant_corpus(uid)
    except RuntimeError as e:
        return f"RAG configuration error: {e}"

    try:
        _ensure_vertex_init(corpus)
        from vertexai import rag  # noqa: PLC0415

        rag_resource = rag.RagResource(rag_corpus=corpus)
        cfg = rag.RagRetrievalConfig(top_k=k)

        filter_kwargs: dict[str, Any] = {}
        if Config.RAG_VECTOR_DISTANCE_THRESHOLD >= 0:
            filter_kwargs["vector_distance_threshold"] = (
                Config.RAG_VECTOR_DISTANCE_THRESHOLD
            )
        if mode == "shared":
            # Apply tenant metadata filter so cross-tenant chunks are excluded.
            key = Config.VERTEX_RAG_TENANT_METADATA_KEY or "tenant_id"
            filter_kwargs["metadata_filter"] = f'{key}="{uid}"'
        # "per_tenant", "from_bq": dedicated corpus per user — no metadata filter.
        if filter_kwargs:
            cfg.filter = rag.Filter(**filter_kwargs)

        resp = rag.retrieval_query(
            text=q,
            rag_resources=[rag_resource],
            rag_retrieval_config=cfg,
        )
        md, chunks = _format_rag_response(resp)
    except Exception as e:
        logger.exception("RAG retrieval failed")
        return f"RAG retrieval error: {e}"

    tool_context.state["rag_chunks"] = chunks
    tool_context.state["rag_corpus"] = corpus
    tool_context.state["rag_mode"] = mode
    tool_context.state["rag_top_k"] = k

    header = (
        f"## RAG corpus retrieval\n"
        f"tenant: `{uid}` | mode: **{mode}** | corpus: `{corpus}` | top_k: {k}"
    )
    if not chunks:
        return f"{header}\n\n{md}"
    return f"{header}\n\n{md}"


# ── Tool 7: score the RAG retrieval as a whole ───────────────────────────────
def score_rag_evidence(
    confidence: float,
    rationale: str,
    tool_context: ToolContext,
) -> str:
    """
    Persist a confidence score in [0.0, 1.0] for the RAG corpus retrieval as a
    whole (one score per query, distinct from per-neighborhood scores).

    Stored in ``tool_context.state['rag_score']`` for use by
    ``synthesize_evidence``.

    Args:
        confidence: Float in [0.0, 1.0] — closer to 1.0 means the RAG chunks
                    strongly address the user's prompt.
        rationale: One- or two-sentence justification.

    Returns:
        Confirmation string with the stored score.
    """
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return "Error: confidence must be a float in [0.0, 1.0]."
    c = max(0.0, min(1.0, c))

    tool_context.state["rag_score"] = {
        "confidence": c,
        "rationale": (rationale or "").strip(),
    }
    logger.info("Scored RAG retrieval → %.3f", c)
    return f"Stored RAG confidence={c:.3f}."


# ── Tool 8: set source weights for synthesis ─────────────────────────────────
def set_source_weights(
    graph_weight: float,
    rag_weight: float,
    rationale: str,
    tool_context: ToolContext,
) -> str:
    """
    Record the agent's chosen weighting of the two evidence sources for the
    final synthesis. Weights are auto-normalised to sum to 1.0 if they don't
    already.

    Guidance:
      * **graph_weight ≈ 1, rag_weight ≈ 0** when the question is about
        relationships, dependencies, or skill demonstration that the KG
        captures explicitly.
      * **graph_weight ≈ 0, rag_weight ≈ 1** when the question needs verbatim
        text, code excerpts, or full-file context only the RAG corpus has.
      * **roughly equal** when both sources contribute different but
        complementary evidence.

    Args:
        graph_weight: Non-negative weight for the Neo4j graph evidence.
        rag_weight: Non-negative weight for the RAG corpus evidence.
        rationale: One short sentence justifying the chosen split.

    Returns:
        Confirmation string with the normalised weights.
    """
    try:
        gw = max(0.0, float(graph_weight))
        rw = max(0.0, float(rag_weight))
    except (TypeError, ValueError):
        return "Error: weights must be non-negative floats."

    total = gw + rw
    if total <= 0:
        return "Error: at least one weight must be > 0."

    gw_n = gw / total
    rw_n = rw / total

    tool_context.state["source_weights"] = {
        "graph": gw_n,
        "rag": rw_n,
        "rationale": (rationale or "").strip(),
    }
    logger.info("Source weights → graph=%.2f, rag=%.2f", gw_n, rw_n)
    return (
        f"Stored source weights: graph={gw_n:.2f}, rag={rw_n:.2f} "
        f"(normalised from {gw:.2f}/{rw:.2f})."
    )


# ── Tool 9: synthesize final two-source evidence ─────────────────────────────
def synthesize_evidence(
    tool_context: ToolContext,
    min_neighborhood_confidence: float = 0.4,
    max_neighborhoods: int = 5,
    max_rag_chunks: int = 6,
) -> str:
    """
    Build the final two-source evidence block combining graph neighborhoods
    and RAG corpus chunks, annotated with the source weights set by
    ``set_source_weights``.

    The agent uses this block as the grounding for its final answer. It does
    NOT itself synthesize a textual answer — that is the agent's job, using
    the structured weighted evidence below.

    Args:
        min_neighborhood_confidence: Drop neighborhoods scored below this.
        max_neighborhoods: Cap on graph neighborhoods included.
        max_rag_chunks: Cap on RAG chunks included.

    Returns:
        A consolidated Markdown block with two sections (graph, rag), each
        weighted by the agent's chosen source weights.
    """
    weights = tool_context.state.get("source_weights")
    if not weights:
        return (
            "Error: call ``set_source_weights`` first so the synthesis knows "
            "how to balance the two sources."
        )

    nbhds: dict[str, dict[str, Any]] = tool_context.state.get("neighborhoods") or {}
    seed_scores: dict[str, dict[str, Any]] = (
        tool_context.state.get("seed_scores") or {}
    )
    rag_chunks: list[dict[str, Any]] = tool_context.state.get("rag_chunks") or []
    rag_score: dict[str, Any] = tool_context.state.get("rag_score") or {}

    # Filter + rank graph neighborhoods
    ranked: list[tuple[str, float, str]] = []
    for seed in nbhds:
        sc = seed_scores.get(seed, {})
        conf = float(sc.get("confidence", 0.0))
        if conf < float(min_neighborhood_confidence):
            continue
        ranked.append((seed, conf, sc.get("rationale", "")))
    ranked.sort(key=lambda x: -x[1])
    ranked = ranked[: int(max_neighborhoods)]

    # Cap RAG chunks (already ranked by RAG Engine score)
    rag_top = rag_chunks[: int(max_rag_chunks)]

    out: list[str] = [
        "# Synthesized two-source evidence",
        f"query: {tool_context.state.get('query', '(unknown)')}",
        f"difficulty: {tool_context.state.get('query_difficulty', '(unknown)')}",
        f"source weights — graph: **{weights['graph']:.2f}**, rag: "
        f"**{weights['rag']:.2f}**",
        f"weight rationale: {weights.get('rationale', '(none)')}",
        "",
    ]

    out.append(f"## Graph evidence (weight {weights['graph']:.2f})")
    if not ranked:
        out.append("_No graph neighborhoods met the confidence threshold._")
    else:
        for seed, conf, rationale in ranked:
            nh = nbhds[seed]
            out.append(f"### Seed `{seed}` — confidence {conf:.2f}")
            if rationale:
                out.append(f"_rationale_: {rationale}")
            out.append(
                f"hop_depth={nh.get('hop_depth')} | "
                f"nodes={len(nh.get('nodes') or [])} | "
                f"edges={len(nh.get('edges') or [])}"
            )
            out.append("**Nodes**:")
            for n in (nh.get("nodes") or [])[:20]:
                text = (n.get("text") or "").replace("\n", " ")
                out.append(f"- `{n['id']}` ({n.get('kind', 'Entity')}) — {text}")
            out.append("**Edges**:")
            for e in (nh.get("edges") or [])[:40]:
                out.append(
                    f"- `{e['source']}` —[{e['type']}]→ `{e['target']}`"
                )
            out.append("")

    out.append("")
    out.append(f"## RAG corpus evidence (weight {weights['rag']:.2f})")
    rag_conf = rag_score.get("confidence")
    if rag_conf is not None:
        out.append(f"RAG retrieval confidence: **{float(rag_conf):.2f}**")
        if rag_score.get("rationale"):
            out.append(f"_rationale_: {rag_score['rationale']}")
    if not rag_top:
        out.append("_No RAG chunks retrieved (or RAG not configured)._")
    else:
        for chunk in rag_top:
            score_s = (
                f"{chunk['score']:.3f}" if chunk.get("score") is not None else "n/a"
            )
            out.append(
                f"### Chunk {chunk['index']} (similarity {score_s})"
            )
            out.append(f"_source_: `{chunk.get('source', '(unknown)')}`")
            text = chunk.get("text", "")
            out.append(text)
            out.append("")

    out.append("## Score summary")
    out.append("```json")
    out.append(
        json.dumps(
            {
                "source_weights": weights,
                "neighborhoods": seed_scores,
                "rag": rag_score,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    out.append("```")

    return "\n".join(out)
