"""
Configuration for the graphRAG agent.

Loads environment variables from KG_agent/.env (parent directory) so this folder
shares Neo4j + Vertex AI credentials with experience_pipeline.py without
duplicating secrets.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a: Any, **_k: Any) -> bool:
        return False


_HERE = Path(__file__).resolve().parent
_KG_AGENT_ROOT = _HERE.parent
_PROJECT_ROOT = _KG_AGENT_ROOT.parent

load_dotenv(_HERE / ".env")
load_dotenv(_KG_AGENT_ROOT / ".env")
load_dotenv(_PROJECT_ROOT / ".env")


def _str(key: str, default: str = "") -> str:
    return (os.environ.get(key, default) or default).strip()


def _int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class Config:
    # ── Neo4j (same instance as experience_pipeline) ─────────────────────────
    # Intentionally no localhost defaults in cloud environments.
    NEO4J_URI: str = _str("NEO4J_URI")
    NEO4J_USER: str = _str("NEO4J_USER")
    NEO4J_PASSWORD: str = _str("NEO4J_PASSWORD")
    NEO4J_DATABASE: str = _str("GRAPHRAG_NEO4J_DATABASE")  # empty = server default

    # ── Vertex AI ────────────────────────────────────────────────────────────
    GOOGLE_CLOUD_PROJECT: str = _str("GOOGLE_CLOUD_PROJECT") or _str("GCP_PROJECT")
    VERTEX_LOCATION: str = _str("VERTEX_LOCATION", "us-central1")

    # ── Embedding (must match experience_pipeline.EXP_EMBEDDING_MODEL) ───────
    EMBEDDING_MODEL: str = _str("EXP_EMBEDDING_MODEL", "text-embedding-004")
    VECTOR_INDEX_NAME: str = _str("EXP_VECTOR_INDEX_NAME", "exp_entity_embedding")

    # ── LLM for the agent itself ─────────────────────────────────────────────
    AGENT_MODEL: str = _str("GRAPHRAG_AGENT_MODEL", "gemini-2.5-flash")

    # ── Retrieval tuning bounds ──────────────────────────────────────────────
    # Difficulty buckets pick (top_k, hop_depth) within these limits.
    MIN_TOP_K: int = _int("GRAPHRAG_MIN_TOP_K", 3)
    MAX_TOP_K: int = _int("GRAPHRAG_MAX_TOP_K", 25)
    MIN_HOPS: int = _int("GRAPHRAG_MIN_HOPS", 1)
    MAX_HOPS: int = _int("GRAPHRAG_MAX_HOPS", 3)

    # Hard caps on what one neighborhood query can return (prevents runaway).
    MAX_NEIGHBOR_NODES: int = _int("GRAPHRAG_MAX_NEIGHBOR_NODES", 60)
    MAX_NEIGHBOR_EDGES: int = _int("GRAPHRAG_MAX_NEIGHBOR_EDGES", 120)

    # ── Vertex AI RAG Engine corpus (per-tenant) ─────────────────────────────
    # Two configuration modes (in priority order):
    #
    #   1. Per-tenant corpus template (preferred — strict isolation):
    #        VERTEX_RAG_CORPUS_TEMPLATE=projects/<PROJ>/locations/<LOC>/ragCorpora/gitai-{tenant}
    #      The agent substitutes the normalised tenant id for {tenant} so each
    #      user's GCS-derived chunks live in a dedicated corpus.
    #
    #   2. Single shared corpus with metadata filter (fallback):
    #        VERTEX_RAG_CORPUS=projects/<PROJ>/locations/<LOC>/ragCorpora/<id>
    #        VERTEX_RAG_TENANT_METADATA_KEY=tenant_id  (default)
    #      Documents must have been uploaded with metadata containing the
    #      tenant id under this key. The agent applies a Filter at query time.
    VERTEX_RAG_CORPUS_TEMPLATE: str = _str("VERTEX_RAG_CORPUS_TEMPLATE")
    VERTEX_RAG_CORPUS: str = _str("VERTEX_RAG_CORPUS")
    VERTEX_RAG_TENANT_METADATA_KEY: str = _str("VERTEX_RAG_TENANT_METADATA_KEY", "tenant_id")
    RAG_TOP_K: int = _int("GRAPHRAG_RAG_TOP_K", 8)
    # Use -1.0 sentinel to mean "no threshold"; positive floats clip results.
    RAG_VECTOR_DISTANCE_THRESHOLD: float = _float("GRAPHRAG_RAG_VECTOR_DISTANCE_THRESHOLD", -1.0)

    @classmethod
    def is_rag_configured(cls) -> bool:
        return bool(cls.VERTEX_RAG_CORPUS_TEMPLATE or cls.VERTEX_RAG_CORPUS)
