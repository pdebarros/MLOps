"""
Experience-focused KG pipeline  —  stand-alone test script.

Purpose
-------
Produce a *richer* knowledge graph than ``prod_pipeline.py`` by combining THREE
evidence sources for every Python file:

  1. The **raw Python source** downloaded directly from GCS.
  2. The **structural summary** already cached by prod_pipeline (if present).
  3. The **technical summary** already cached by prod_pipeline (if present).

Three-phase construction
------------------------
Phase 1 — Experience Assessment
    An LLM (same Gemini / Vertex backend as the rest of the pipeline) reads all
    three sources and writes a structured "experience assessment": a labelled list
    of Concepts, Patterns, Skills, Frameworks, Algorithms, and Paradigms that the
    file *demonstrably* uses — not just imports.  This assessment is cached in GCS
    under ``summaries_experience/`` with its own ingestion flag
    (``exp_kg_ingested``) so it never touches prod_pipeline's caches.

Phase 2 — Structural graph extraction  (STRUCTURAL pass)
    An ``LLMGraphTransformer`` with an *enhanced* structural schema extracts
    Module / Class / Function / API / Config / Exception nodes plus richer
    relationship types (RAISES, HANDLES, INSTANTIATES, DECORATES, READS_FROM,
    WRITES_TO, …).  Input = experience assessment + technical summary + raw source.

Phase 3 — Experience graph extraction  (EXPERIENCE pass)
    A second ``LLMGraphTransformer`` with an experience schema extracts
    Concept / Pattern / Skill / Framework / Algorithm / Paradigm nodes linked to
    the File via DEMONSTRATES, APPLIES, IMPLEMENTS, COMBINES.

Both passes write to the same Neo4j database with full multi-tenancy (label +
property isolation identical to prod_pipeline).

Isolation guarantees
--------------------
* No existing file is modified.
* Experience summaries use GCS prefix  ``summaries_experience/`` (not
  ``summaries_structural/`` or ``summaries_technical/``).
* The ingestion flag ``exp_kg_ingested`` is distinct from ``kg_ingested``.
* The two graph transformers are instantiated locally — they do NOT share the
  singleton in ``tools2.py``.

Run
---
  cd KG_agent/
  python experience_pipeline.py --user-id u_123
  python experience_pipeline.py --user-id u_123 --neo4j-database neo4j \\
      --structural-batch-size 2 --experience-batch-size 1 \\
      --max-source-chars 4000 --force
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any

from config import Config
from google.cloud import storage
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase
from pipeline2 import (
    code_prefix_for_user,
    load_code_files_for_names,
    load_summary_record,
    normalize_user_id,
    run_parallel_summaries,
    summary_json_exists,
    upload_summary_record,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("Orchestrator.experience_pipeline")

# ── GCS folder names ────────────────────────────────────────────────────────
SUMMARIES_STRUCTURAL = "summaries_structural"
SUMMARIES_TECHNICAL = "summaries_technical"
SUMMARIES_EXPERIENCE = "summaries_experience"

# ── Ingestion tracking (separate from prod_pipeline) ────────────────────────
EXP_INGESTION_FLAG = "exp_kg_ingested"
EXP_INGESTION_PIPELINE = "exp_kg_ingested_by"
EXP_INGESTION_TS = "exp_kg_ingested_at"
EXP_PIPELINE_NAME = "experience_pipeline_v1"

# ── Enhanced structural schema ───────────────────────────────────────────────
STRUCTURAL_NODE_TYPES: list[str] = [
    "Module",
    "Class",
    "Function",
    "Variable",
    "Dependency",
    "Service",
    "Database",
    "File",
    "API",
    "Config",
    "Exception",
]

STRUCTURAL_REL_TYPES: list[str] = [
    "IMPORTS",
    "CALLS",
    "DEFINES",
    "INHERITS",
    "USES",
    "DEPENDS_ON",
    "INTERACTS_WITH",
    "REFERENCES",
    "RAISES",
    "HANDLES",
    "INSTANTIATES",
    "DECORATES",
    "OVERRIDES",
    "RETURNS",
    "READS_FROM",
    "WRITES_TO",
    "CONFIGURED_BY",
    "EXTENDS",
    "TRIGGERS",
    "VALIDATES",
]

STRUCTURAL_EXTRA_INSTRUCTIONS = (
    "Extract code architecture: Module, Class, Function, API, Config, Exception nodes and "
    "their relationships.\n"
    "Relationship guidance:\n"
    "  IMPORTS — module-level import dependency\n"
    "  CALLS — function/method invocation\n"
    "  DEFINES — a Module/Class defines a Function or nested Class\n"
    "  INHERITS — class hierarchy\n"
    "  INSTANTIATES — object construction (ClassName())\n"
    "  RAISES — function raises an Exception\n"
    "  HANDLES — try/except catching an Exception\n"
    "  DECORATES — decorator applied to a function or class\n"
    "  READS_FROM — reading from a file, stream, or DB\n"
    "  WRITES_TO — writing to a file, stream, or DB\n"
    "  CONFIGURED_BY — component configured by a Config node\n"
    "  RETURNS — function returns a specific type/object\n"
    "Always include a File node representing the source file. "
    "Use precise, canonical names (e.g., 'Config' not 'the configuration object')."
)

# ── Experience schema ────────────────────────────────────────────────────────
EXPERIENCE_NODE_TYPES: list[str] = [
    "Concept",
    "Pattern",
    "Skill",
    "Framework",
    "Algorithm",
    "Paradigm",
    "File",
]

EXPERIENCE_REL_TYPES: list[str] = [
    "DEMONSTRATES",
    "APPLIES",
    "IMPLEMENTS",
    "USES_ADVANCED",
    "USES_BASIC",
    "COMBINES",
    "REQUIRES_KNOWLEDGE_OF",
    "IS_INSTANCE_OF",
    "RELATED_TO",
]

EXPERIENCE_EXTRA_INSTRUCTIONS = (
    "Extract programming experience nodes from the experience assessment below.\n"
    "Node types:\n"
    "  Concept   — specific language feature (AsyncAwait, TypeHinting, ContextManager)\n"
    "  Pattern   — design/architectural pattern (DependencyInjection, RetryLogic, Singleton)\n"
    "  Skill     — broad competency (ErrorHandling, Concurrency, APIDesign, DataValidation)\n"
    "  Framework — non-trivial framework usage (FastAPI, LangChain, Neo4j, Pydantic)\n"
    "  Algorithm — implemented algorithm (BatchProcessing, IncrementalIngestion, BinarySearch)\n"
    "  Paradigm  — programming paradigm (FunctionalProgramming, AsyncProgramming, OOP)\n"
    "  File      — the source file being analysed\n"
    "Relationship guidance:\n"
    "  DEMONSTRATES     — File demonstrates knowledge of a Concept/Pattern/Skill\n"
    "  APPLIES          — File/Function applies a Pattern or Framework concretely\n"
    "  IMPLEMENTS       — Concept/Pattern implements an Algorithm\n"
    "  USES_ADVANCED    — sophisticated, non-obvious framework/concept usage\n"
    "  USES_BASIC       — boilerplate or simple usage of a framework/concept\n"
    "  COMBINES         — two Concepts/Patterns are used together in a sophisticated way\n"
    "  REQUIRES_KNOWLEDGE_OF — one Concept presupposes understanding of another\n"
    "Use PascalCase canonical names. "
    "Only claim DEMONSTRATES/USES_ADVANCED if the code shows actual application, "
    "not just an import statement."
)

# ── Experience assessor prompts ──────────────────────────────────────────────
EXPERIENCE_ASSESSOR_INSTRUCTION = """\
You are a senior software engineering assessor evaluating a developer's demonstrated \
programming knowledge from Python source code.

CRITICAL RULES:
  1. Only report knowledge that is ACTIVELY demonstrated in the implementation —
     not merely imported or available in the environment.
  2. Be specific and canonical with names (e.g. "AsyncContextManager" not "context
     manager"; "DependencyInjection" not "dependency injection pattern").
  3. Rate sophistication honestly:
       basic     — boilerplate / copy-paste level usage
       competent — understands the tool and uses it correctly
       advanced  — non-obvious, sophisticated usage showing deep understanding
  4. Focus on what a hiring engineer could conclude about the developer's knowledge.
  5. Ignore library imports that are never meaningfully called or extended.

Output format — one finding per line:
  CATEGORY: CanonicalPascalCaseName | sophistication | one-sentence evidence

Categories: CONCEPT, PATTERN, SKILL, FRAMEWORK, ALGORITHM, PARADIGM

Examples:
  CONCEPT: AsyncAwait | advanced | Uses asyncio.gather() for parallel LLM calls with proper cancellation
  PATTERN: DependencyInjection | competent | FastAPI Depends() for config and auth service injection
  SKILL: ErrorHandling | advanced | Multi-level exception handling with custom error types and fallback chains
  FRAMEWORK: LangChain | advanced | Custom chain composition with LLMGraphTransformer and Neo4j integration
  ALGORITHM: TokenEstimation | basic | Simple char-count heuristic (len/4) for token approximation
  PARADIGM: FunctionalProgramming | competent | Heavy use of list comprehensions and functools patterns\
"""

EXPERIENCE_USER_PREFIX = """\
FILE: {file_name}

=== STRUCTURAL SUMMARY (architecture view) ===
{structural_summary}

=== TECHNICAL SUMMARY (implementation view) ===
{technical_summary}

=== PYTHON SOURCE ===
{raw_source}

Produce the experience assessment for this file following the format above.\
"""

EXPERIENCE_FOCUS_SUFFIX = (
    "\n\nBe rigorous: if you cannot find clear evidence in the code for a claim, omit it. "
    "Quality over quantity."
)


# ── Environment-driven hyperparameter defaults ───────────────────────────────
def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


DEFAULT_STRUCTURAL_BATCH = _env_int("EXP_STRUCTURAL_KG_BATCH", 2)
DEFAULT_STRUCTURAL_OVERLAP = _env_int("EXP_STRUCTURAL_KG_OVERLAP", 1)
DEFAULT_STRUCTURAL_MAX_DOC_CHARS = _env_int("EXP_STRUCTURAL_MAX_DOC_CHARS", 3000)
DEFAULT_EXPERIENCE_BATCH = _env_int("EXP_EXPERIENCE_KG_BATCH", 1)
DEFAULT_EXPERIENCE_OVERLAP = _env_int("EXP_EXPERIENCE_KG_OVERLAP", 0)
DEFAULT_EXPERIENCE_MAX_DOC_CHARS = _env_int("EXP_EXPERIENCE_MAX_DOC_CHARS", 2500)
DEFAULT_MAX_SOURCE_CHARS = _env_int("EXP_MAX_SOURCE_CHARS", 4000)

# ── Vector index / embedding config ─────────────────────────────────────────
EMBEDDING_MODEL = os.environ.get("EXP_EMBEDDING_MODEL", "text-embedding-004")
EMBEDDING_DIMENSIONS = _env_int("EXP_EMBEDDING_DIMENSIONS", 768)
EMBEDDING_BATCH_SIZE = _env_int("EXP_EMBEDDING_BATCH_SIZE", 250)
VECTOR_INDEX_NAME = os.environ.get("EXP_VECTOR_INDEX_NAME", "exp_entity_embedding")


# ── LLM / transformer factories ──────────────────────────────────────────────
def _build_llm():
    """Build a Vertex AI / Gemini LLM (or HF) from Config — same logic as tools2."""
    backend = Config.KG_GRAPH_BACKEND.strip().lower()
    if backend in ("vertex", "gemini", "gcp"):
        from langchain_google_vertexai import ChatVertexAI

        kwargs: dict[str, Any] = {
            "model": Config.VERTEX_GEMINI_MODEL,
            "location": Config.VERTEX_LOCATION,
            "temperature": Config.VERTEX_TEMPERATURE,
        }
        if Config.GOOGLE_CLOUD_PROJECT:
            kwargs["project"] = Config.GOOGLE_CLOUD_PROJECT
        if Config.VERTEX_MAX_TOKENS is not None:
            kwargs["max_tokens"] = int(Config.VERTEX_MAX_TOKENS)
        logger.info(
            "[exp] LLM: Vertex AI %s @ %s (project=%s)",
            Config.VERTEX_GEMINI_MODEL,
            Config.VERTEX_LOCATION,
            Config.GOOGLE_CLOUD_PROJECT or "(ADC default)",
        )
        return ChatVertexAI(**kwargs)

    if backend in ("huggingface", "hf", "local"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers import pipeline as hf_pipeline
        from langchain_huggingface import HuggingFacePipeline

        device = "mps" if torch.backends.mps.is_available() else (
            0 if torch.cuda.is_available() else -1
        )
        dtype = torch.float16 if device != -1 else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(Config.HF_GRAPH_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            Config.HF_GRAPH_MODEL,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        pipe = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=Config.HF_MAX_NEW_TOKENS,
            return_full_text=False,
        )
        return HuggingFacePipeline(pipeline=pipe, model_id=Config.HF_GRAPH_MODEL)

    raise ValueError(f"Unknown KG_GRAPH_BACKEND={backend!r}; use 'vertex' or 'huggingface'.")


def _build_structural_transformer(llm) -> LLMGraphTransformer:
    use_hf = Config.KG_GRAPH_BACKEND.strip().lower() in ("huggingface", "hf", "local")
    ignore = use_hf or Config.KG_VERTEX_IGNORE_TOOL_USAGE
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=STRUCTURAL_NODE_TYPES,
        allowed_relationships=STRUCTURAL_REL_TYPES,
        strict_mode=False,
        ignore_tool_usage=ignore,
        additional_instructions=STRUCTURAL_EXTRA_INSTRUCTIONS,
    )


def _build_experience_transformer(llm) -> LLMGraphTransformer:
    use_hf = Config.KG_GRAPH_BACKEND.strip().lower() in ("huggingface", "hf", "local")
    ignore = use_hf or Config.KG_VERTEX_IGNORE_TOOL_USAGE
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=EXPERIENCE_NODE_TYPES,
        allowed_relationships=EXPERIENCE_REL_TYPES,
        strict_mode=False,
        ignore_tool_usage=ignore,
        additional_instructions=EXPERIENCE_EXTRA_INSTRUCTIONS,
    )


# ── Neo4j persistence (self-contained, no singleton from tools2) ─────────────
_neo4j_driver = None


def _get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            Config.NEO4J_URI, auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
    return _neo4j_driver


def _sanitize_tenant_label(uid: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", uid)
    if safe and safe[0].isdigit():
        safe = "U" + safe
    return f"User_{safe}"


def _node_text(node_id: str, node_type: str, props: dict[str, Any]) -> str:
    """
    Build a concise, informative text string stored as ``n.text`` on every node.
    This is the field embedded for vector similarity search in GraphRAG.

    Format:  "<NodeType>: <Name> [<sophistication>] — <evidence/description>"
    The sophistication and evidence fields are optional and come from whatever
    properties the LLMGraphTransformer extracted.
    """
    header = f"{node_type}: {node_id}"

    soph = props.get("sophistication") or props.get("level")
    if soph and isinstance(soph, str) and soph.strip():
        header += f" [{soph.strip()}]"

    details: list[str] = []
    for field in ("evidence", "description", "summary", "detail", "purpose"):
        val = props.get(field)
        if val and isinstance(val, str) and len(val.strip()) > 4:
            details.append(val.strip())
            break

    cat = props.get("category")
    if cat and isinstance(cat, str) and cat.strip().lower() not in ("", node_type.lower()):
        details.append(f"category: {cat.strip()}")

    if details:
        return f"{header} — {' | '.join(details)}"
    return header


def _persist_graph_documents(
    graph_documents: list,
    *,
    extra_node_props: dict[str, Any] | None = None,
    database: str | None = None,
    tenant_id: str | None = None,
) -> tuple[int, int]:
    """Persist graph documents to Neo4j with multi-tenancy (same logic as tools2)."""
    driver = _get_neo4j_driver()
    nodes_written = 0
    edges_written = 0
    session_kwargs: dict[str, Any] = {}
    if database:
        session_kwargs["database"] = database

    user_label = _sanitize_tenant_label(tenant_id) if tenant_id else None

    with driver.session(**session_kwargs) as session:
        for gd in graph_documents:
            for node in gd.nodes:
                node_id = str(node.id)
                node_type = str(node.type or "Entity")
                props = dict(node.properties or {})
                props["kind"] = node_type
                merged = dict(props)
                if extra_node_props:
                    merged.update(extra_node_props)
                # Always set a human-readable text field for GraphRAG embedding.
                # Built from whatever descriptive properties the LLM extracted so
                # the embedding captures semantic meaning, not just the entity name.
                merged["text"] = _node_text(node_id, node_type, props)

                if tenant_id:
                    merged["tenantId"] = tenant_id
                    session.run(
                        f"""
                        MERGE (n:Entity:`{user_label}` {{id: $id, tenantId: $tid}})
                        SET n += $props
                        """,
                        id=node_id,
                        tid=tenant_id,
                        props=merged,
                    )
                else:
                    session.run(
                        "MERGE (n:Entity {id: $id}) SET n += $props",
                        id=node_id,
                        props=merged,
                    )
                nodes_written += 1

            for rel in gd.relationships:
                src_id = str(rel.source.id)
                dst_id = str(rel.target.id)
                rel_type = str(rel.type or "RELATED_TO")
                rel_props = dict(rel.properties or {})
                rel_props["type"] = rel_type

                if tenant_id:
                    rel_props["tenantId"] = tenant_id
                    session.run(
                        f"""
                        MATCH (a:Entity:`{user_label}` {{id: $src, tenantId: $tid}})
                        MATCH (b:Entity:`{user_label}` {{id: $dst, tenantId: $tid}})
                        MERGE (a)-[r:REL {{type: $rtype, tenantId: $tid}}]->(b)
                        SET r += $rprops
                        """,
                        src=src_id,
                        dst=dst_id,
                        rtype=rel_type,
                        tid=tenant_id,
                        rprops=rel_props,
                    )
                else:
                    session.run(
                        """
                        MATCH (a:Entity {id: $src})
                        MATCH (b:Entity {id: $dst})
                        MERGE (a)-[r:REL {type: $rtype}]->(b)
                        SET r += $rprops
                        """,
                        src=src_id,
                        dst=dst_id,
                        rtype=rel_type,
                        rprops=rel_props,
                    )
                edges_written += 1

    return nodes_written, edges_written


# ── Vector index + embedding helpers ─────────────────────────────────────────
_embedding_model = None


def _get_embedding_model():
    """Lazy-init the Vertex AI text-embedding model (uses same ADC credentials)."""
    global _embedding_model
    if _embedding_model is None:
        from langchain_google_vertexai import VertexAIEmbeddings  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "model_name": EMBEDDING_MODEL,
            "location": Config.VERTEX_LOCATION,
        }
        if Config.GOOGLE_CLOUD_PROJECT:
            kwargs["project"] = Config.GOOGLE_CLOUD_PROJECT
        logger.info(
            "[exp-embed] Embedding model: %s @ %s (project=%s)",
            EMBEDDING_MODEL,
            Config.VERTEX_LOCATION,
            Config.GOOGLE_CLOUD_PROJECT or "(ADC default)",
        )
        _embedding_model = VertexAIEmbeddings(**kwargs)
    return _embedding_model


def _ensure_vector_index(database: str | None = None) -> None:
    """
    Create the Neo4j vector index on ``Entity.embedding`` if it doesn't exist.

    Uses ``IF NOT EXISTS`` so it is safe to call on every pipeline run.
    Dimensions and similarity function are configurable via env vars
    ``EXP_EMBEDDING_DIMENSIONS`` (default 768) and the index is named by
    ``EXP_VECTOR_INDEX_NAME`` (default ``exp_entity_embedding``).
    """
    driver = _get_neo4j_driver()
    session_kwargs: dict[str, Any] = {}
    if database:
        session_kwargs["database"] = database

    with driver.session(**session_kwargs) as session:
        session.run(
            f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (n:Entity)
            ON (n.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMENSIONS},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )
    logger.info(
        "[exp-embed] Vector index '%s' ready (%d dims, cosine similarity)",
        VECTOR_INDEX_NAME,
        EMBEDDING_DIMENSIONS,
    )


def _embed_tenant_nodes(
    tenant_id: str,
    database: str | None = None,
    *,
    batch_size: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Compute and store embeddings for all tenant nodes that have a ``text``
    property but no ``embedding`` yet (or all nodes when ``force=True``).

    Embeddings are written back to Neo4j as ``n.embedding`` (float list) so
    the vector index can serve approximate nearest-neighbour lookups for
    GraphRAG hybrid retrieval.

    Returns a summary dict with ``embedded``, ``failed``, and ``skipped`` counts.
    """
    bs = batch_size if batch_size is not None else EMBEDDING_BATCH_SIZE
    driver = _get_neo4j_driver()
    session_kwargs: dict[str, Any] = {}
    if database:
        session_kwargs["database"] = database

    user_label = _sanitize_tenant_label(tenant_id)
    where_clause = (
        "WHERE n.text IS NOT NULL"
        if force
        else "WHERE n.text IS NOT NULL AND n.embedding IS NULL"
    )

    with driver.session(**session_kwargs) as session:
        result = session.run(
            f"""
            MATCH (n:Entity:`{user_label}` {{tenantId: $tid}})
            {where_clause}
            RETURN n.id AS node_id, n.text AS text
            """,
            tid=tenant_id,
        )
        rows = [(r["node_id"], r["text"]) for r in result if r["text"]]

    if not rows:
        logger.info("[exp-embed] No nodes need embedding (tenant=%s)", tenant_id)
        return {"embedded": 0, "failed": 0, "skipped": 0}

    logger.info(
        "[exp-embed] Embedding %d node(s) for tenant=%s (batch_size=%d) …",
        len(rows), tenant_id, bs,
    )

    model = _get_embedding_model()
    embedded = 0
    failed = 0
    total_batches = (len(rows) + bs - 1) // bs

    for batch_idx in range(total_batches):
        chunk = rows[batch_idx * bs: (batch_idx + 1) * bs]
        node_ids = [r[0] for r in chunk]
        texts = [r[1] for r in chunk]

        try:
            vectors = model.embed_documents(texts)
            with driver.session(**session_kwargs) as session:
                for node_id, vector in zip(node_ids, vectors):
                    session.run(
                        f"""
                        MATCH (n:Entity:`{user_label}` {{id: $nid, tenantId: $tid}})
                        SET n.embedding = $emb
                        """,
                        nid=node_id,
                        tid=tenant_id,
                        emb=vector,
                    )
            embedded += len(chunk)
            logger.info(
                "[exp-embed] Batch %d/%d — stored %d embedding(s)",
                batch_idx + 1, total_batches, len(chunk),
            )
        except Exception as exc:
            logger.error("[exp-embed] Batch %d/%d failed: %s", batch_idx + 1, total_batches, exc)
            failed += len(chunk)

    return {"embedded": embedded, "failed": failed, "skipped": 0}


# ── GCS helpers ──────────────────────────────────────────────────────────────
def _exp_blob_path(code_prefix: str, code_blob_name: str) -> str:
    """GCS path for the experience assessment JSON for one .py file."""
    if not code_blob_name.startswith(code_prefix):
        raise ValueError(f"{code_blob_name!r} must start with {code_prefix!r}")
    rel = code_blob_name[len(code_prefix):]
    return f"{code_prefix}{SUMMARIES_EXPERIENCE}/{rel}.json"


def _prod_blob_path(code_prefix: str, code_blob_name: str, track_folder: str) -> str:
    """GCS path for a structural or technical summary cached by prod_pipeline."""
    if not code_blob_name.startswith(code_prefix):
        raise ValueError(f"{code_blob_name!r} must start with {code_prefix!r}")
    rel = code_blob_name[len(code_prefix):]
    return f"{code_prefix}{track_folder}/{rel}.json"


def _list_python_blob_names(bucket: storage.Bucket, code_prefix: str) -> list[str]:
    """List .py blobs, skipping all summary sub-folders."""
    skip = {
        f"{code_prefix}{SUMMARIES_STRUCTURAL}/",
        f"{code_prefix}{SUMMARIES_TECHNICAL}/",
        f"{code_prefix}{SUMMARIES_EXPERIENCE}/",
    }
    out: list[str] = []
    for blob in bucket.list_blobs(prefix=code_prefix):
        name = blob.name
        if any(name.startswith(s) for s in skip):
            continue
        if name.endswith(".py"):
            out.append(name)
    return sorted(out)


def _load_prod_summary_text(
    bucket: storage.Bucket, code_prefix: str, blob_name: str, track_folder: str
) -> str:
    """Return the summary text from a prod_pipeline cache, or empty string if missing."""
    path = _prod_blob_path(code_prefix, blob_name, track_folder)
    rec = load_summary_record(bucket, path)
    if rec and rec.get("status") == "completed" and rec.get("summary"):
        return str(rec["summary"])
    return ""


def _exp_is_ingested(bucket: storage.Bucket, code_prefix: str, blob_name: str) -> bool:
    path = _exp_blob_path(code_prefix, blob_name)
    if not summary_json_exists(bucket, path):
        return False
    rec = load_summary_record(bucket, path)
    return bool(rec and rec.get(EXP_INGESTION_FLAG))


def _exp_has_assessment(bucket: storage.Bucket, code_prefix: str, blob_name: str) -> bool:
    path = _exp_blob_path(code_prefix, blob_name)
    if not summary_json_exists(bucket, path):
        return False
    rec = load_summary_record(bucket, path)
    return bool(rec and rec.get("status") == "completed" and rec.get("summary"))


def _mark_exp_ingested(
    bucket: storage.Bucket, code_prefix: str, blob_name: str, rec: dict[str, Any]
) -> None:
    path = _exp_blob_path(code_prefix, blob_name)
    out = dict(rec)
    out[EXP_INGESTION_FLAG] = True
    out[EXP_INGESTION_PIPELINE] = EXP_PIPELINE_NAME
    out[EXP_INGESTION_TS] = datetime.now(timezone.utc).isoformat()
    upload_summary_record(bucket, path, out)


def _load_exp_assessment_text(
    bucket: storage.Bucket, code_prefix: str, blob_name: str
) -> str:
    path = _exp_blob_path(code_prefix, blob_name)
    rec = load_summary_record(bucket, path)
    if rec and rec.get("status") == "completed" and rec.get("summary"):
        return str(rec["summary"])
    return ""


# ── Phase 1: experience assessment generation ────────────────────────────────
async def _generate_experience_assessments(
    *,
    bucket: storage.Bucket,
    code_prefix: str,
    files_needing_assessment: list[str],
    raw_by_name: dict[str, str],
    model_name: str,
    max_parallel: int,
    uid: str,
    max_source_chars: int,
) -> None:
    """
    Generate experience assessments for the given files and cache them in GCS.

    ``raw_by_name`` maps blob_name → raw source text.
    Structural + technical summaries are fetched from GCS if available.
    """
    if not files_needing_assessment:
        return

    # Build the combined context for each file that needs assessment
    file_data: list[dict[str, str]] = []
    for blob_name in files_needing_assessment:
        raw_src = raw_by_name.get(blob_name, "(source not available)")
        if max_source_chars > 0 and len(raw_src) > max_source_chars:
            raw_src = raw_src[:max_source_chars] + "\n... [truncated]"

        struct_txt = _load_prod_summary_text(
            bucket, code_prefix, blob_name, SUMMARIES_STRUCTURAL
        )
        tech_txt = _load_prod_summary_text(
            bucket, code_prefix, blob_name, SUMMARIES_TECHNICAL
        )

        # Construct the combined input document
        content = EXPERIENCE_USER_PREFIX.format(
            file_name=blob_name,
            structural_summary=struct_txt or "(not yet generated — run prod_pipeline first)",
            technical_summary=tech_txt or "(not yet generated — run prod_pipeline first)",
            raw_source=raw_src,
        )
        file_data.append({"file_name": blob_name, "file_content": content})

    adk_user = f"exp-pipeline-assess-{uid}"
    logger.info(
        "[exp-assess] Generating assessments for %d file(s) …", len(file_data)
    )
    results = await run_parallel_summaries(
        file_data,
        model_name,
        max_parallel,
        adk_user,
        summarizer_instruction=EXPERIENCE_ASSESSOR_INSTRUCTION,
        summary_focus_suffix=EXPERIENCE_FOCUS_SUFFIX,
    )

    for res in results:
        blob_name = res["file"]
        path = _exp_blob_path(code_prefix, blob_name)
        upload_summary_record(
            bucket,
            path,
            {
                "file": blob_name,
                "summary": res.get("summary"),
                "status": res.get("status", "failed"),
                "error": res.get("error"),
                "track": "experience",
            },
        )
        logger.info("[exp-assess] Cached assessment → gs://%s/%s", bucket.name, path)


# ── Document builders ─────────────────────────────────────────────────────────
def _build_structural_doc(
    blob_name: str,
    assessment: str,
    tech_summary: str,
    raw_src: str,
    max_source_chars: int,
) -> Document:
    """Build the rich input document for the structural graph extraction pass."""
    if max_source_chars > 0 and len(raw_src) > max_source_chars:
        raw_src = raw_src[:max_source_chars] + "\n... [truncated]"

    parts = [f"FILE: {blob_name}"]
    if assessment:
        parts.append(f"EXPERIENCE ASSESSMENT:\n{assessment}")
    if tech_summary:
        parts.append(f"TECHNICAL IMPLEMENTATION:\n{tech_summary}")
    if raw_src:
        parts.append(f"PYTHON SOURCE:\n{raw_src}")

    return Document(
        page_content="\n\n---\n\n".join(parts),
        metadata={"file": blob_name},
    )


def _build_experience_doc(blob_name: str, assessment: str) -> Document:
    """Build the input document for the experience graph extraction pass."""
    content = f"FILE: {blob_name}\n\nEXPERIENCE ASSESSMENT:\n{assessment}"
    return Document(page_content=content, metadata={"file": blob_name})


def _chunk_docs(docs: list[Document], size: int, overlap: int) -> list[list[Document]]:
    if size <= 0:
        size = 5
    overlap = max(0, min(overlap, size - 1))
    chunks: list[list[Document]] = []
    step = size - overlap
    start = 0
    while start < len(docs):
        chunks.append(docs[start: start + size])
        start += step
    return chunks


def _batch_documents(chunks: list[list[Document]]) -> list[Document]:
    batch_docs: list[Document] = []
    for idx, chunk in enumerate(chunks):
        merged = "\n\n---\n\n".join(d.page_content for d in chunk)
        files = [d.metadata.get("file") for d in chunk]
        batch_docs.append(
            Document(page_content=merged, metadata={"batch": idx, "files": files})
        )
    return batch_docs


# ── Phase 2+3: transformer extraction + Neo4j write ──────────────────────────
def _run_transformer_pass(
    transformer: LLMGraphTransformer,
    docs: list[Document],
    batch_size: int,
    batch_overlap: int,
    max_doc_chars: int,
    pass_label: str,
) -> list:
    """Run transformer over batched docs, return graph_documents list."""
    capped: list[Document] = []
    for d in docs:
        text = d.page_content
        if max_doc_chars > 0 and len(text) > max_doc_chars:
            text = text[:max_doc_chars]
        capped.append(Document(page_content=text, metadata=d.metadata))

    chunks = _chunk_docs(capped, batch_size, batch_overlap)
    batches = _batch_documents(chunks)
    logger.info(
        "[%s] %d doc(s) → %d batch(es)", pass_label, len(docs), len(batches)
    )

    all_gd: list = []
    for i, batch in enumerate(batches, start=1):
        logger.info("[%s] Transforming batch %d/%d …", pass_label, i, len(batches))
        try:
            gd = transformer.convert_to_graph_documents([batch])
            all_gd.extend(gd)
            n = sum(len(g.nodes) for g in gd)
            e = sum(len(g.relationships) for g in gd)
            logger.info("[%s] batch %d → %d nodes, %d edges", pass_label, i, n, e)
        except Exception as exc:
            logger.error("[%s] batch %d failed: %s", pass_label, i, exc)

    return all_gd


# ── Main pipeline ─────────────────────────────────────────────────────────────
async def run_experience_pipeline(
    user_id: str,
    *,
    neo4j_database: str | None = None,
    structural_batch_size: int | None = None,
    structural_batch_overlap: int | None = None,
    structural_max_doc_chars: int | None = None,
    experience_batch_size: int | None = None,
    experience_batch_overlap: int | None = None,
    experience_max_doc_chars: int | None = None,
    max_source_chars: int | None = None,
    force: bool = False,
    skip_embeddings: bool = False,
) -> dict[str, Any]:
    """
    Full experience pipeline for one user.

    Parameters
    ----------
    user_id          : raw user ID (will be normalised)
    neo4j_database   : Neo4j database name; None = AuraDB default
    structural_*     : hyperparams for the structural graph extraction pass
    experience_*     : hyperparams for the experience graph extraction pass
    max_source_chars : max raw Python source chars included per document
    force            : if True, re-assess and re-ingest files already processed
    skip_embeddings  : if True, skip Phase 4 vector embedding (graph still written)
    """
    uid = normalize_user_id(user_id)
    tenant_id = uid
    code_prefix = code_prefix_for_user(uid)
    model_name = Config.GEMINI_MODEL
    max_parallel = Config.SUMMARY_MAX_PARALLEL

    s_batch = structural_batch_size if structural_batch_size is not None else DEFAULT_STRUCTURAL_BATCH
    s_overlap = structural_batch_overlap if structural_batch_overlap is not None else DEFAULT_STRUCTURAL_OVERLAP
    s_chars = structural_max_doc_chars if structural_max_doc_chars is not None else DEFAULT_STRUCTURAL_MAX_DOC_CHARS
    e_batch = experience_batch_size if experience_batch_size is not None else DEFAULT_EXPERIENCE_BATCH
    e_overlap = experience_batch_overlap if experience_batch_overlap is not None else DEFAULT_EXPERIENCE_OVERLAP
    e_chars = experience_max_doc_chars if experience_max_doc_chars is not None else DEFAULT_EXPERIENCE_MAX_DOC_CHARS
    src_chars = max_source_chars if max_source_chars is not None else DEFAULT_MAX_SOURCE_CHARS

    logger.info(
        "[exp] START  user=%s  tenant=%s  bucket=%s  db=%s  force=%s",
        uid, tenant_id, Config.GCS_BUCKET_NAME, neo4j_database or "(default)", force,
    )

    client = storage.Client()
    bucket = client.bucket(Config.GCS_BUCKET_NAME)

    py_blob_names = _list_python_blob_names(bucket, code_prefix)
    if not py_blob_names:
        return {"status": "skipped", "reason": "no_py_files", "user_id": uid}

    logger.info("[exp] Found %d .py file(s)", len(py_blob_names))

    # ── Phase 1: determine which files need new experience assessments ────────
    needs_assessment = [
        name for name in py_blob_names
        if force or not _exp_has_assessment(bucket, code_prefix, name)
    ]

    if needs_assessment:
        raw_file_data = load_code_files_for_names(bucket, needs_assessment)
        raw_by_name = {d["file_name"]: d["file_content"] for d in raw_file_data}
        await _generate_experience_assessments(
            bucket=bucket,
            code_prefix=code_prefix,
            files_needing_assessment=needs_assessment,
            raw_by_name=raw_by_name,
            model_name=model_name,
            max_parallel=max_parallel,
            uid=uid,
            max_source_chars=src_chars,
        )
    else:
        logger.info("[exp] All assessments already cached (use --force to regenerate).")
        raw_by_name = {}

    # ── Determine files ready for KG ingestion ────────────────────────────────
    ready_for_kg = [
        name for name in py_blob_names
        if _exp_has_assessment(bucket, code_prefix, name)
        and (force or not _exp_is_ingested(bucket, code_prefix, name))
    ]

    if not ready_for_kg:
        return {
            "status": "completed",
            "user_id": uid,
            "message": "All files already ingested. Use --force to re-ingest.",
            "py_file_count": len(py_blob_names),
        }

    logger.info("[exp] %d file(s) ready for KG ingestion", len(ready_for_kg))

    # ── Load all needed raw source (for structural docs) ─────────────────────
    names_needing_raw = [n for n in ready_for_kg if n not in raw_by_name]
    if names_needing_raw:
        extra = load_code_files_for_names(bucket, names_needing_raw)
        raw_by_name.update({d["file_name"]: d["file_content"] for d in extra})

    # ── Build documents for both passes ──────────────────────────────────────
    structural_docs: list[Document] = []
    experience_docs: list[Document] = []

    for blob_name in ready_for_kg:
        assessment = _load_exp_assessment_text(bucket, code_prefix, blob_name)
        tech_txt = _load_prod_summary_text(bucket, code_prefix, blob_name, SUMMARIES_TECHNICAL)
        raw_src = raw_by_name.get(blob_name, "")

        if not assessment:
            logger.warning("[exp] No assessment for %s — skipping", blob_name)
            continue

        structural_docs.append(
            _build_structural_doc(blob_name, assessment, tech_txt, raw_src, src_chars)
        )
        experience_docs.append(_build_experience_doc(blob_name, assessment))

    if not structural_docs:
        return {
            "status": "error",
            "user_id": uid,
            "message": "No assessment documents could be built.",
        }

    # ── Build the two LLM transformers (separate from tools2 singletons) ──────
    logger.info("[exp] Loading LLM backend: %s …", Config.KG_GRAPH_BACKEND)
    llm = _build_llm()
    structural_transformer = _build_structural_transformer(llm)
    experience_transformer = _build_experience_transformer(llm)

    # ── Phase 2: structural graph extraction ──────────────────────────────────
    logger.info("[exp] Phase 2 — Structural graph extraction …")
    structural_gd = _run_transformer_pass(
        structural_transformer,
        structural_docs,
        s_batch,
        s_overlap,
        s_chars,
        "struct",
    )

    struct_nodes = 0
    struct_edges = 0
    struct_error: str | None = None

    if structural_gd:
        try:
            struct_nodes, struct_edges = _persist_graph_documents(
                structural_gd,
                extra_node_props={"kg_track": "experience_structural", "exp_pipeline": True},
                database=neo4j_database,
                tenant_id=tenant_id,
            )
            logger.info(
                "[exp] Structural: wrote %d nodes, %d edges", struct_nodes, struct_edges
            )
        except Exception as exc:
            struct_error = str(exc)
            logger.error("[exp] Structural Neo4j write failed: %s", exc)
    else:
        struct_error = "No graph documents produced by structural transformer"
        logger.warning("[exp] %s", struct_error)

    # ── Phase 3: experience graph extraction ──────────────────────────────────
    logger.info("[exp] Phase 3 — Experience graph extraction …")
    experience_gd = _run_transformer_pass(
        experience_transformer,
        experience_docs,
        e_batch,
        e_overlap,
        e_chars,
        "experience",
    )

    exp_nodes = 0
    exp_edges = 0
    exp_error: str | None = None

    if experience_gd:
        try:
            exp_nodes, exp_edges = _persist_graph_documents(
                experience_gd,
                extra_node_props={"kg_track": "experience", "exp_pipeline": True},
                database=neo4j_database,
                tenant_id=tenant_id,
            )
            logger.info(
                "[exp] Experience: wrote %d nodes, %d edges", exp_nodes, exp_edges
            )
        except Exception as exc:
            exp_error = str(exc)
            logger.error("[exp] Experience Neo4j write failed: %s", exc)
    else:
        exp_error = "No graph documents produced by experience transformer"
        logger.warning("[exp] %s", exp_error)

    # ── Mark files as ingested (only if BOTH passes succeeded) ────────────────
    both_ok = struct_error is None and exp_error is None
    if both_ok and not force:
        for blob_name in ready_for_kg:
            path = _exp_blob_path(code_prefix, blob_name)
            rec = load_summary_record(bucket, path) or {
                "file": blob_name, "status": "completed", "summary": "",
            }
            _mark_exp_ingested(bucket, code_prefix, blob_name, rec)

    # ── Phase 4: vector index + embeddings ────────────────────────────────────
    embed_report: dict[str, Any] = {"skipped": True}
    if not skip_embeddings:
        logger.info("[exp] Phase 4 — Vector index + embeddings …")
        try:
            _ensure_vector_index(neo4j_database)
        except Exception as exc:
            logger.warning("[exp] Vector index creation failed (non-fatal): %s", exc)

        if struct_nodes > 0 or exp_nodes > 0:
            try:
                embed_report = _embed_tenant_nodes(
                    tenant_id,
                    neo4j_database,
                    batch_size=EMBEDDING_BATCH_SIZE,
                    force=force,
                )
                logger.info(
                    "[exp] Embeddings: %d stored, %d failed",
                    embed_report.get("embedded", 0),
                    embed_report.get("failed", 0),
                )
            except Exception as exc:
                logger.error("[exp] Embedding phase failed (non-fatal): %s", exc)
                embed_report = {"embedded": 0, "failed": -1, "error": str(exc)}
        else:
            logger.info("[exp] No new nodes written — embedding phase skipped.")
            embed_report = {"embedded": 0, "failed": 0, "skipped": 0}
    else:
        logger.info("[exp] Embedding phase skipped (--skip-embeddings).")

    report = {
        "status": "completed" if both_ok else "partial_error",
        "user_id": uid,
        "tenant_id": tenant_id,
        "py_file_count": len(py_blob_names),
        "files_assessed_now": len(needs_assessment),
        "files_sent_to_kg": len(ready_for_kg),
        "structural_pass": {
            "nodes_written": struct_nodes,
            "edges_written": struct_edges,
            "error": struct_error,
        },
        "experience_pass": {
            "nodes_written": exp_nodes,
            "edges_written": exp_edges,
            "error": exp_error,
        },
        "embedding_pass": embed_report,
        "vector_index": VECTOR_INDEX_NAME,
        "hyperparameters": {
            "structural": {
                "batch_size": s_batch,
                "batch_overlap": s_overlap,
                "max_doc_chars": s_chars,
            },
            "experience": {
                "batch_size": e_batch,
                "batch_overlap": e_overlap,
                "max_doc_chars": e_chars,
            },
            "max_source_chars": src_chars,
        },
        "marked_ingested": both_ok and not force,
    }
    logger.info("[exp] DONE: %s", json.dumps(report, indent=2))
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Experience-focused KG pipeline — test script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--user-id", required=True, help="Raw user ID (will be normalised)")
    p.add_argument("--neo4j-database", default=None, help="Neo4j database name (omit for AuraDB default)")
    p.add_argument("--structural-batch-size", type=int, default=None)
    p.add_argument("--structural-batch-overlap", type=int, default=None)
    p.add_argument("--structural-max-doc-chars", type=int, default=None)
    p.add_argument("--experience-batch-size", type=int, default=None)
    p.add_argument("--experience-batch-overlap", type=int, default=None)
    p.add_argument("--experience-max-doc-chars", type=int, default=None)
    p.add_argument(
        "--max-source-chars",
        type=int,
        default=None,
        help="Max raw Python source characters included per document (0 = unlimited)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-generate assessments and re-ingest even for files already processed",
    )
    p.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip Phase 4 (vector index + embedding computation). Graph is still written.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    result = asyncio.run(
        run_experience_pipeline(
            args.user_id,
            neo4j_database=args.neo4j_database,
            structural_batch_size=args.structural_batch_size,
            structural_batch_overlap=args.structural_batch_overlap,
            structural_max_doc_chars=args.structural_max_doc_chars,
            experience_batch_size=args.experience_batch_size,
            experience_batch_overlap=args.experience_batch_overlap,
            experience_max_doc_chars=args.experience_max_doc_chars,
            max_source_chars=args.max_source_chars,
            force=args.force,
            skip_embeddings=args.skip_embeddings,
        )
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
