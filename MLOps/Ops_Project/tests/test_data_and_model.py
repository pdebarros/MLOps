"""
Ten focused tests for **data conventions** (user ids, GCS paths, KG schema config) and
**weak eval model** helpers (read-only Cypher lint, fenced Cypher extraction, property allowlist).

No live Neo4j, GCS, or LLM calls — safe for CI.

Run from ``Ops_Project``::

  pytest tests/test_data_and_model.py -v
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_OPS = Path(__file__).resolve().parent.parent

# --- eval_agent (data paths / Vertex corpus parsing) ---
_EA = _OPS / "eval_agent"
if str(_EA) not in sys.path:
    sys.path.insert(0, str(_EA))

from config import (  # noqa: E402
    normalize_user_id,
    project_and_location_for_vertex,
    scoring_blob_path_for_user,
)

# KG_agent also has ``config.py`` — load it explicitly so it does not clash with eval_agent.config.
_kg_spec = importlib.util.spec_from_file_location(
    "mlops_kg_agent_config", _OPS / "KG_agent" / "config.py"
)
assert _kg_spec and _kg_spec.loader
_kg_mod = importlib.util.module_from_spec(_kg_spec)
_kg_spec.loader.exec_module(_kg_mod)
KGConfig = _kg_mod.Config

# --- Weak student model (Cypher safety + parsing) ---
# ``model.py`` uses ``from .config import config``. Importing the real ``eval_agent``
# package runs ``__init__.py`` → agent → tools → Neo4j ``Driver`` (heavy / stub mismatch).
# Load only ``config.py`` + ``model.py`` under a synthetic package name.
_weak_pkg = "_ops_weak_eval_agent"
_weak_root = _OPS / "Weak_Model" / "eval_agent"
_pkg = types.ModuleType(_weak_pkg)
_pkg.__path__ = [str(_weak_root)]  # type: ignore[attr-defined]
sys.modules[_weak_pkg] = _pkg

_weak_cfg_spec = importlib.util.spec_from_file_location(
    f"{_weak_pkg}.config", _weak_root / "config.py"
)
assert _weak_cfg_spec and _weak_cfg_spec.loader
_weak_cfg = importlib.util.module_from_spec(_weak_cfg_spec)
sys.modules[f"{_weak_pkg}.config"] = _weak_cfg
_weak_cfg_spec.loader.exec_module(_weak_cfg)

_weak_model_spec = importlib.util.spec_from_file_location(
    f"{_weak_pkg}.model", _weak_root / "model.py"
)
assert _weak_model_spec and _weak_model_spec.loader
weak_model = importlib.util.module_from_spec(_weak_model_spec)
sys.modules[f"{_weak_pkg}.model"] = weak_model
_weak_model_spec.loader.exec_module(weak_model)


# ----- Data (5) -----


def test_data_normalize_user_id_strips_whitespace_and_slashes():
    assert normalize_user_id("  u_demo/  ") == "u_demo"


def test_data_normalize_user_id_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        normalize_user_id("   ")
    with pytest.raises(ValueError, match="non-empty"):
        normalize_user_id("//")


def test_data_scoring_blob_path_for_user():
    assert scoring_blob_path_for_user("u_42") == "u_42/scoring"


def test_data_project_and_location_from_vertex_corpus_resource():
    corpus = "projects/my-proj/locations/europe-west1/ragCorpora/abc123"
    proj, loc = project_and_location_for_vertex(corpus)
    assert proj == "my-proj"
    assert loc == "europe-west1"


def test_data_kg_config_schema_lists_populated():
    assert "Module" in KGConfig.ALLOWED_NODE_TYPES
    assert "Function" in KGConfig.ALLOWED_NODE_TYPES
    assert "DEPENDS_ON" in KGConfig.ALLOWED_REL_TYPES
    assert isinstance(KGConfig.HF_KG_MODEL, str) and KGConfig.HF_KG_MODEL


# ----- Model / weak Cypher helpers (5) -----


def test_model_validate_read_only_accepts_match():
    ok, reason = weak_model._validate_read_only_cypher(
        "MATCH (n:Person) RETURN n LIMIT 5"
    )
    assert ok is True
    assert reason == ""


def test_model_validate_read_only_rejects_create():
    ok, reason = weak_model._validate_read_only_cypher(
        "CREATE (n:Person {name: 'x'}) RETURN n"
    )
    assert ok is False
    assert "forbidden" in reason.lower()


def test_model_validate_read_only_rejects_multiple_statements():
    ok, reason = weak_model._validate_read_only_cypher(
        "MATCH (n) RETURN n; MATCH (m) RETURN m"
    )
    assert ok is False
    assert "multiple" in reason.lower()


def test_model_extract_cypher_block_from_markdown_fence():
    text = """Here is the query:
```cypher
MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 3
```
"""
    block = weak_model._extract_cypher_block(text)
    assert block is not None
    assert block.startswith("MATCH")
    assert "LIMIT 3" in block


def test_model_lint_cypher_property_keys_respects_allowlist():
    allowed = frozenset({"name", "id"})
    ok, _ = weak_model._lint_cypher_property_keys(
        "MATCH (n:Person) WHERE n.name = 'Ann' RETURN n.id",
        allowed,
    )
    assert ok is True

    bad_ok, bad_reason = weak_model._lint_cypher_property_keys(
        "MATCH (n:Person) RETURN n.unknown_prop",
        allowed,
    )
    assert bad_ok is False
    assert "unknown_prop" in bad_reason
