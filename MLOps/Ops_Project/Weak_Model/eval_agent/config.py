"""Eval agent settings (Vertex RAG Engine corpus + shared GCP env)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_a: Any, **_k: Any) -> bool:
        return False


_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT.parent / ".env")


def _env_str(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    if v is None or v == "":
        return default
    return v


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return int(raw)


def _env_float(key: str) -> float | None:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    return float(raw)


def google_cloud_project() -> str | None:
    return (
        _env_str("GOOGLE_CLOUD_PROJECT")
        or _env_str("GCP_PROJECT")
        or _env_str("GOOGLE_CLOUD_PROJECT_ID")
    )


_CORPUS_FULL_RE = re.compile(
    r"^projects/(?P<proj>[^/]+)/locations/(?P<loc>[^/]+)/ragCorpora/(?P<id>[^/]+)$"
)


def resolve_rag_corpus_resource() -> str | None:
    """
    Corpus resource name for Vertex RAG Engine.

    Set either:
      VERTEX_RAG_CORPUS=projects/.../locations/.../ragCorpora/...
    or
      RAG_CORPUS_ID=<id> plus GOOGLE_CLOUD_PROJECT and VERTEX_LOCATION (or GOOGLE_CLOUD_REGION).
    """
    full = (_env_str("VERTEX_RAG_CORPUS") or "").strip()
    if full:
        return full
    cid = (_env_str("RAG_CORPUS_ID") or "").strip()
    if not cid:
        return None
    proj = google_cloud_project()
    if not proj:
        return None
    loc = (
        _env_str("VERTEX_LOCATION")
        or _env_str("GOOGLE_CLOUD_REGION")
        or "us-central1"
    )
    return f"projects/{proj}/locations/{loc}/ragCorpora/{cid}"


def project_and_location_for_vertex(corpus_resource: str) -> tuple[str, str]:
    """Derive (project_id, location) for vertexai.init from the corpus name."""
    m = _CORPUS_FULL_RE.match(corpus_resource.strip())
    if m:
        return m.group("proj"), m.group("loc")
    proj = google_cloud_project()
    loc = (
        _env_str("VERTEX_LOCATION")
        or _env_str("GOOGLE_CLOUD_REGION")
        or "us-central1"
    )
    if not proj:
        raise ValueError(
            "Set VERTEX_RAG_CORPUS to a full resource name, or set GOOGLE_CLOUD_PROJECT "
            "and VERTEX_LOCATION with RAG_CORPUS_ID."
        )
    return proj, loc


class Config:
    # Neo4j (same env vars as KG_agent)
    NEO4J_URI: str = _env_str("NEO4J_URI", "bolt://localhost:7687") or "bolt://localhost:7687"
    NEO4J_USER: str = _env_str("NEO4J_USER", "neo4j") or "neo4j"
    NEO4J_PASSWORD: str = _env_str("NEO4J_PASSWORD", "cowboy78910") or "cowboy78910"

    # Vertex RAG Engine (evaluator grounding; higher defaults = broader retrieval per query)
    VERTEX_RAG_CORPUS: str | None = resolve_rag_corpus_resource()
    RAG_TOP_K: int = _env_int("RAG_TOP_K", 16)
    RAG_TOP_K_MAX: int = _env_int("RAG_TOP_K_MAX", 48)
    RAG_VECTOR_DISTANCE_THRESHOLD: float | None = _env_float("RAG_VECTOR_DISTANCE_THRESHOLD")

    # Neighborhood sampling (kg_reconstruct_test-style)
    EVAL_DEFAULT_K: int = _env_int("EVAL_DEFAULT_K", 2)
    EVAL_MAX_STARTS: int = _env_int("EVAL_MAX_STARTS", 3)
    EVAL_ENTITY_LIST_LIMIT: int = _env_int("EVAL_ENTITY_LIST_LIMIT", 20)

    # GCS append-only scoring log (JSONL) — same bucket layout as pipeline2: gs://<bucket>/<user_id>/...
    GCS_BUCKET_NAME: str = _env_str("GCS_BUCKET_NAME", "codebases-04-03-26") or "codebases-04-03-26"
    GCS_SCORING_FILENAME: str = _env_str("GCS_SCORING_FILENAME", "scoring") or "scoring"

    # Weak Cypher student (Groq OpenAI-compatible API; see eval_agent/model.py)
    GROQ_API_KEY: str | None = _env_str("GROQ_API_KEY") or _env_str("GROQ_KEY")
    GROQ_BASE_URL: str = (
        _env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1") or "https://api.groq.com/openai/v1"
    )
    # Default: Llama 3.1 8B Instant on Groq (https://console.groq.com/docs/model/llama-3.1-8b-instant)
    GROQ_MODEL: str = _env_str("GROQ_MODEL", "llama-3.1-8b-instant") or "llama-3.1-8b-instant"
    WEAK_CYPHER_MAX_ROWS: int = _env_int("WEAK_CYPHER_MAX_ROWS", 80)
    # Optional path to append one JSON object per weak_model_query run (Groq + Cypher trace).
    WEAK_CYPHER_TRACE_LOG: str | None = _env_str("WEAK_CYPHER_TRACE_LOG")
    WEAK_CYPHER_TRACE_MAX_TEXT: int = _env_int("WEAK_CYPHER_TRACE_MAX_TEXT", 8000)

    # Gemini multi-step Cypher explorer (eval_agent/model2.py)
    GEMINI_CYPHER_TRACE_LOG: str | None = _env_str("GEMINI_CYPHER_TRACE_LOG")
    GEMINI_CYPHER_TRACE_MAX_TEXT: int = _env_int("GEMINI_CYPHER_TRACE_MAX_TEXT", 8000)


config = Config()


def normalize_user_id(user_id: str) -> str:
    """Same convention as KG_agent/pipeline2.py: non-empty folder name under the bucket."""
    uid = user_id.strip().strip("/")
    if not uid:
        raise ValueError("user_id must be non-empty")
    return uid


def scoring_blob_path_for_user(user_id: str) -> str:
    """Object name: ``<user_id>/scoring`` (NDJSON append log per user)."""
    uid = normalize_user_id(user_id)
    fn = (config.GCS_SCORING_FILENAME or "scoring").strip() or "scoring"
    return f"{uid}/{fn}"
