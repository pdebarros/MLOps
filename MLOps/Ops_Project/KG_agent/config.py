"""
Central configuration loaded from the environment (and optional `.env` files).

Import this module to apply `load_dotenv` for `KG_agent/.env` then project-root `.env`.

Typical variables (set in `.env` or the shell):
  Neo4j: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
  GCS / pipeline: GCS_BUCKET_NAME, GCS_PREFIX, GEMINI_MODEL, SUMMARY_MAX_PARALLEL
  Vertex / Gemini (KG + reconstruct test): GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION,
    VERTEX_LOCATION, VERTEX_GEMINI_MODEL, GEMINI_GRAPH_MODEL, VERTEX_TEMPERATURE,
    VERTEX_MAX_TOKENS
  KG extraction (tools2): KG_GRAPH_BACKEND, HF_*, KG_MAX_DOC_CHARS, KG_BATCH_*,
    ALLOWED_NODE_TYPES, ALLOWED_REL_TYPES, KG_VERTEX_IGNORE_TOOL_USAGE
  Legacy tools.py: HF_KG_MODEL
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


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    return float(raw)


def _env_bool_flag(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes")


def _comma_list(key: str, default_csv: str) -> list[str]:
    raw = os.environ.get(key, default_csv)
    return [x.strip() for x in raw.split(",") if x.strip()]


def _vertex_max_tokens() -> int | None:
    raw = os.environ.get("VERTEX_MAX_TOKENS")
    if raw is None or str(raw).strip() == "":
        return None
    return int(raw)


def google_cloud_project() -> str | None:
    """GCP project id from common env aliases."""
    return (
        _env_str("GOOGLE_CLOUD_PROJECT")
        or _env_str("GCP_PROJECT")
        or _env_str("GOOGLE_CLOUD_PROJECT_ID")
    )


class Config:
    """Read-only settings snapshot (evaluated at import time)."""
    # --- Neo4j (tools2, tools.py, kg_reconstruct_test) ---
    #NEO4J_URI: str = _env_str("NEO4J_URI", "neo4j+s://76522009.databases.neo4j.io") or "neo4j+s://76522009.databases.neo4j.io"
    #NEO4J_USER: str = _env_str("NEO4J_USER", "76522009") or "76522009"
    #NEO4J_PASSWORD: str = _env_str("NEO4J_PASSWORD", "ClEp-qPfGT4SiXaGCtNwhMW-NKqcwWPUEyfZAMtUABw") or "ClEp-qPfGT4SiXaGCtNwhMW-NKqcwWPUEyfZAMtUABw"


    # for local config
    NEO4J_URI: str = _env_str("NEO4J_URI", "bolt://localhost:7687") or "bolt://localhost:7687"
    NEO4J_USER: str = _env_str("NEO4J_USER", "neo4j") or "neo4j"
    NEO4J_PASSWORD: str = _env_str("NEO4J_PASSWORD", "cowboy78910") or "cowboy78910"
    # --- GCS + summarization (pipeline2) ---
    GCS_BUCKET_NAME: str = _env_str("GCS_BUCKET_NAME", "codebases-04-03-26") or "codebases-04-03-26"
    GCS_PREFIX: str | None = _env_str("GCS_PREFIX") or None
    GEMINI_MODEL: str = _env_str("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-pro"
    SUMMARY_MAX_PARALLEL: int = _env_int("SUMMARY_MAX_PARALLEL", 4)

    # --- KG backend (tools2) ---
    KG_GRAPH_BACKEND: str = (
        (_env_str("KG_GRAPH_BACKEND", "vertex") or "vertex").strip().lower()
    )

    # --- Vertex AI / Gemini (tools2, kg_reconstruct_test) ---
    VERTEX_GEMINI_MODEL: str = (
        _env_str("VERTEX_GEMINI_MODEL")
        or _env_str("GEMINI_GRAPH_MODEL")
        or "gemini-2.5-pro"
    )
    GOOGLE_CLOUD_PROJECT: str | None = google_cloud_project()
    VERTEX_LOCATION: str = (
        _env_str("GOOGLE_CLOUD_REGION")
        or _env_str("VERTEX_LOCATION")
        or "us-central1"
    ) or "us-central1"
    VERTEX_TEMPERATURE: float = _env_float("VERTEX_TEMPERATURE", 0.3)
    VERTEX_MAX_TOKENS: int | None = _vertex_max_tokens()
    KG_VERTEX_IGNORE_TOOL_USAGE: bool = _env_bool_flag("KG_VERTEX_IGNORE_TOOL_USAGE")

    # --- Hugging Face graph model (tools2) ---
    HF_GRAPH_MODEL: str = _env_str("HF_GRAPH_MODEL", "Qwen/Qwen2.5-3B-Instruct") or (
        "Qwen/Qwen2.5-3B-Instruct"
    )
    HF_MAX_NEW_TOKENS: int = _env_int("HF_MAX_NEW_TOKENS", 256)
    HF_TEMPERATURE: float = _env_float("HF_TEMPERATURE", 0.3)

    # --- KG batching / truncation (tools2) ---
    KG_MAX_DOC_CHARS: int = _env_int("KG_MAX_DOC_CHARS", 1200)
    KG_BATCH_SIZE: int = _env_int("KG_BATCH_SIZE", 1)
    KG_BATCH_OVERLAP: int = _env_int("KG_BATCH_OVERLAP", 0)

    # --- KG schema (tools2) ---
    ALLOWED_NODE_TYPES: list[str] = _comma_list(
        "ALLOWED_NODE_TYPES",
        "Module,Class,Function,Variable,Dependency,Service,Database,File,Other",
    )
    ALLOWED_REL_TYPES: list[str] = _comma_list(
        "ALLOWED_REL_TYPES",
        "DEPENDS_ON,IMPORTS,CALLS,USES,DEFINES,INHERITS,INTERACTS_WITH,REFERENCES",
    )

    # --- Legacy tools.py HF model ---
    HF_KG_MODEL: str = _env_str("HF_KG_MODEL", "google/flan-t5-large") or "google/flan-t5-large"


# Singleton-style access (same as class attributes)
config = Config()
