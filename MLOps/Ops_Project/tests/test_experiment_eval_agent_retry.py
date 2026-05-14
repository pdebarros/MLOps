from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

_OPS = Path(__file__).resolve().parent.parent
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

if "mlflow" not in sys.modules:
    sys.modules["mlflow"] = types.ModuleType("mlflow")
if "pipeline2" not in sys.modules:
    pipeline2_stub = types.ModuleType("pipeline2")
    pipeline2_stub.normalize_user_id = lambda x: x
    sys.modules["pipeline2"] = pipeline2_stub
if "pipeline3" not in sys.modules:
    pipeline3_stub = types.ModuleType("pipeline3")
    sys.modules["pipeline3"] = pipeline3_stub

import experiment  # noqa: E402


def _install_eval_agent_stub() -> None:
    pkg = types.ModuleType("eval_agent")
    agent_mod = types.ModuleType("eval_agent.agent")
    agent_mod.root_agent = object()
    pkg.agent = agent_mod
    sys.modules["eval_agent"] = pkg
    sys.modules["eval_agent.agent"] = agent_mod


def test_run_eval_agent_session_retries_once_on_silent_response(monkeypatch):
    _install_eval_agent_stub()

    class FakeRunner:
        run_calls = 0

        def __init__(self, **_kwargs):
            pass

        def run_async(self, **_kwargs):
            FakeRunner.run_calls += 1
            return object()

    monkeypatch.setattr(sys.modules["google.adk.runners"], "Runner", FakeRunner)

    calls = {"n": 0}

    async def fake_final_text(_events_source):
        calls["n"] += 1
        return "" if calls["n"] == 1 else "final answer"

    monkeypatch.setattr(experiment, "_final_text_from_events", fake_final_text)

    out = asyncio.run(experiment.run_eval_agent_session("u_1", "db_1"))
    assert out == "final answer"
    assert calls["n"] == 2
    assert FakeRunner.run_calls == 2


def test_run_eval_agent_session_no_retry_when_first_response_has_text(monkeypatch):
    _install_eval_agent_stub()

    class FakeRunner:
        run_calls = 0

        def __init__(self, **_kwargs):
            pass

        def run_async(self, **_kwargs):
            FakeRunner.run_calls += 1
            return object()

    monkeypatch.setattr(sys.modules["google.adk.runners"], "Runner", FakeRunner)

    async def fake_final_text(_events_source):
        return "ready"

    monkeypatch.setattr(experiment, "_final_text_from_events", fake_final_text)

    out = asyncio.run(experiment.run_eval_agent_session("u_1", "db_1"))
    assert out == "ready"
    assert FakeRunner.run_calls == 1
