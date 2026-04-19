"""
Pytest coverage for pipeline3 (dual-track KG: structural + technical).

Runs without real GCS, Gemini, or Neo4j — uses mocks for orchestration paths.

Placed under tests/ (not KG_agent/) so collection does not import KG_agent.__init__,
which pulls optional ADK deps.

Run from Ops_Project:

  pytest tests/test_pipeline3.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_OPS = Path(__file__).resolve().parent.parent
_KG = _OPS / "KG_agent"
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

import pipeline3  # noqa: E402


def _track_kwargs_base() -> dict:
    """Minimal valid kwargs for run_single_track (Neo4j / GCS mocked)."""
    return {
        "user_id": "u_ci",
        "bucket": MagicMock(),
        "bucket_name": "test-bucket",
        "code_prefix": "u_ci/",
        "py_blob_names": ["u_ci/pkg/mod.py"],
        "model_name": "gemini-test",
        "max_parallel": 2,
        "track_key": "structural",
        "summary_folder": pipeline3.SUMMARIES_STRUCTURAL,
        "summarizer_instruction": pipeline3.STRUCTURAL_SUMMARIZER_INSTRUCTION,
        "summary_focus_suffix": pipeline3.STRUCTURAL_FOCUS_SUFFIX,
        "user_message_prefix": None,
        "kg_batch_size": 2,
        "kg_batch_overlap": 1,
        "kg_max_doc_chars": 1000,
        "extra_entity_props": {"kg_track_structural": True},
        "neo4j_database": None,
    }


# --- 1–2: summary path mapping (pure) ---


def test_summary_blob_path_track_builds_structural_json_path():
    p = pipeline3.summary_blob_path_track(
        "u_1/", "u_1/src/foo.py", pipeline3.SUMMARIES_STRUCTURAL
    )
    assert p == "u_1/summaries_structural/src/foo.py.json"


def test_summary_blob_path_track_rejects_non_prefixed_blob():
    with pytest.raises(ValueError, match="must start with"):
        pipeline3.summary_blob_path_track(
            "u_1/", "other/src/foo.py", pipeline3.SUMMARIES_TECHNICAL
        )


# --- 3–5: cached summary loading ---


def test_load_cached_results_returns_rows_when_all_complete(monkeypatch):
    bucket = MagicMock()
    names = ["u_1/a.py"]
    rec = {
        "file": "u_1/a.py",
        "summary": "ok",
        "status": "completed",
    }
    expected_path = pipeline3.summary_blob_path_track(
        "u_1/", "u_1/a.py", pipeline3.SUMMARIES_STRUCTURAL
    )

    def load_rec(b, path):
        assert b is bucket
        assert path == expected_path
        return rec

    monkeypatch.setattr(pipeline3, "load_summary_record", load_rec)
    out = pipeline3.load_cached_results_for_kg_track(
        bucket, "u_1/", names, pipeline3.SUMMARIES_STRUCTURAL
    )
    assert out is not None
    assert len(out) == 1
    assert out[0]["summary"] == "ok"
    assert out[0]["status"] == "completed"


def test_load_cached_results_returns_none_when_any_summary_missing(monkeypatch):
    bucket = MagicMock()

    def load_rec(_b, _path):
        return None

    monkeypatch.setattr(pipeline3, "load_summary_record", load_rec)
    out = pipeline3.load_cached_results_for_kg_track(
        bucket, "u_1/", ["u_1/x.py"], pipeline3.SUMMARIES_TECHNICAL
    )
    assert out is None


def test_load_cached_results_returns_none_when_status_not_completed(monkeypatch):
    bucket = MagicMock()

    def load_rec(_b, _path):
        return {"file": "u_1/a.py", "summary": "x", "status": "failed"}

    monkeypatch.setattr(pipeline3, "load_summary_record", load_rec)
    out = pipeline3.load_cached_results_for_kg_track(
        bucket, "u_1/", ["u_1/a.py"], pipeline3.SUMMARIES_STRUCTURAL
    )
    assert out is None


# --- 6: missing-summary filter ---


def test_py_files_missing_summaries_track_only_lists_need_regen(monkeypatch):
    bucket = MagicMock()

    def needs(_b, _prefix, name, folder):
        return name.endswith("need.py")

    monkeypatch.setattr(pipeline3, "needs_new_summary_track", needs)
    names = ["u_1/ok.py", "u_1/need.py"]
    miss = pipeline3.py_files_missing_summaries_track(
        bucket, "u_1/", names, pipeline3.SUMMARIES_STRUCTURAL
    )
    assert miss == ["u_1/need.py"]


# --- 7–8: full pipeline orchestration (async) ---


@pytest.mark.asyncio
async def test_run_pipeline3_skipped_when_no_python_blobs(monkeypatch):
    mock_bucket = MagicMock()
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    monkeypatch.setattr(pipeline3.storage, "Client", lambda: mock_client)
    monkeypatch.setattr(pipeline3, "list_python_blob_names", lambda b, p: [])

    out = await pipeline3.run_pipeline3("u_empty")
    assert out["status"] == "skipped"
    assert out["reason"] == "no_py_files"
    assert out["user_id"] == "u_empty"


@pytest.mark.asyncio
async def test_run_pipeline3_completes_with_both_tracks(monkeypatch):
    mock_bucket = MagicMock()
    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    monkeypatch.setattr(pipeline3.storage, "Client", lambda: mock_client)
    monkeypatch.setattr(
        pipeline3,
        "list_python_blob_names",
        lambda b, p: ["u_1/x.py"],
    )

    async def fake_track(**kwargs):
        return {
            "track": kwargs["track_key"],
            "status": "completed",
            "summary_records": 1,
            "kg_result": {"ok": True},
        }

    monkeypatch.setattr(pipeline3, "run_single_track", fake_track)

    out = await pipeline3.run_pipeline3("u_1", neo4j_database="kg_test_db")
    assert out["status"] == "completed"
    assert out["user_id"] == "u_1"
    assert out["py_file_count"] == 1
    assert out["neo4j_database"] == "kg_test_db"
    assert out["structural"]["status"] == "completed"
    assert out["technical"]["status"] == "completed"


# --- 9: single-track error path (GCS code missing) ---


@pytest.mark.asyncio
async def test_run_single_track_returns_error_when_code_load_fails(monkeypatch):
    monkeypatch.setattr(
        pipeline3,
        "py_files_missing_summaries_track",
        lambda *a, **k: ["u_ci/pkg/mod.py"],
    )
    monkeypatch.setattr(pipeline3, "load_code_files_for_names", lambda b, names: [])

    kw = _track_kwargs_base()
    out = await pipeline3.run_single_track(**kw)
    assert out["status"] == "error"
    assert out["reason"] == "code_load_failed"
    assert out["track"] == "structural"
