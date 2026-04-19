"""
Stub optional ML/GCP deps so KG_agent.pipeline3 can import in CI without the full stack.

pipeline3 → pipeline2 pulls google.adk; pipeline3 → tools2 pulls langchain + neo4j.
Tests mock runtime behavior; they only need importable modules.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _ensure(name: str) -> types.ModuleType:
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)
    return sys.modules[name]


def _stub_heavy_kg_imports() -> None:
    # google.adk (pipeline2)
    ga = _ensure("google.adk.agents.llm_agent")
    ga.Agent = MagicMock()
    ga.ParallelAgent = MagicMock()
    _ensure("google.adk.runners").Runner = MagicMock()
    _ensure(
        "google.adk.sessions.in_memory_session_service"
    ).InMemorySessionService = MagicMock()
    _ensure("google.genai").types = MagicMock()

    # langchain / neo4j (tools2)
    _ensure("langchain_core.documents").Document = MagicMock()
    _ensure("langchain_core.language_models").BaseLanguageModel = MagicMock()
    _ensure("langchain_experimental.graph_transformers").LLMGraphTransformer = MagicMock()
    _ensure("neo4j").GraphDatabase = MagicMock()


_stub_heavy_kg_imports()
