"""
HTTP API for cursor-gitai-bridge: runs the same Node ``dist/cli.js`` used by the CLI.

Start (from ``cursor-gitai-bridge/``)::

    export CURSOR_API_KEY=...
    pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8765
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BRIDGE_ROOT = Path(__file__).resolve().parent.parent
CLI_JS = BRIDGE_ROOT / "dist" / "cli.js"


def _node_exe() -> str:
    n = shutil.which("node")
    if not n:
        raise HTTPException(
            status_code=503,
            detail="`node` not found on PATH; install Node.js (>=20).",
        )
    return n


def _ensure_cli_built() -> None:
    if not CLI_JS.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"Missing {CLI_JS}. Run `npm install && npm run build` in {BRIDGE_ROOT}.",
        )


def _run_cli(args: list[str], *, timeout_sec: int | None) -> subprocess.CompletedProcess[str]:
    _ensure_cli_built()
    cmd = [_node_exe(), str(CLI_JS), *args]
    env = os.environ.copy()
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
            cwd=str(BRIDGE_ROOT),
        )
    except subprocess.TimeoutExpired as e:
        raise HTTPException(
            status_code=504,
            detail=f"CLI timed out after {timeout_sec}s: {' '.join(cmd)}",
        ) from e


watch_proc: subprocess.Popen[str] | None = None
watch_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    async with watch_lock:
        global watch_proc
        if watch_proc is not None and watch_proc.poll() is None:
            watch_proc.terminate()
            try:
                watch_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                watch_proc.kill()
            watch_proc = None


app = FastAPI(
    title="cursor-gitai-bridge API",
    description="Expose watch / ask / session via HTTP (wraps Node CLI).",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    ok = CLI_JS.is_file() and shutil.which("node")
    return {"status": "ok" if ok else "degraded", "cli_js_exists": CLI_JS.is_file(), "node_on_path": bool(shutil.which("node"))}


class AskRequest(BaseModel):
    cwd: str = Field(..., description="Repository root for local Cursor agent")
    question: str = Field(..., min_length=1)
    timeout_sec: int = Field(900, ge=30, le=7200)


class SessionRequest(BaseModel):
    cwd: str
    question: str = Field(..., min_length=1)
    agent_id: str | None = None
    new: bool = False
    data_dir: str = Field("data", description="Where last-agent.json lives (relative to bridge cwd or absolute)")
    timeout_sec: int = Field(900, ge=30, le=7200)


class WatchStartRequest(BaseModel):
    file: str = Field(
        ...,
        description="Path to cursor-proposed-changes.ndjson (or GITAI_LOG_FILE target)",
    )
    out_dir: str = Field("data")
    from_start: bool = False


@app.post("/v1/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    if not os.environ.get("CURSOR_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="CURSOR_API_KEY is not set in the server environment.",
        )
    cwd = str(Path(req.cwd).expanduser().resolve())
    proc = _run_cli(
        ["ask", "--cwd", cwd, req.question],
        timeout_sec=req.timeout_sec,
    )
    return _cli_result(proc)


@app.post("/v1/session")
def session(req: SessionRequest) -> dict[str, Any]:
    if not os.environ.get("CURSOR_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="CURSOR_API_KEY is not set in the server environment.",
        )
    cwd = str(Path(req.cwd).expanduser().resolve())
    parts = ["session", "--cwd", cwd, "--data-dir", req.data_dir]
    if req.new:
        parts.append("--new")
    if req.agent_id:
        parts.extend(["--agent-id", req.agent_id])
    parts.append(req.question)
    proc = _run_cli(parts, timeout_sec=req.timeout_sec)
    return _cli_result(proc)


def _cli_result(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    out = {
        "returncode": proc.returncode,
        "stdout": proc.stdout or "",
        "stderr": proc.stderr or "",
    }
    if proc.returncode != 0:
        raise HTTPException(status_code=502, detail=out)
    return out


@app.post("/v1/watch/start")
async def watch_start(req: WatchStartRequest) -> dict[str, Any]:
    global watch_proc
    async with watch_lock:
        if watch_proc is not None and watch_proc.poll() is None:
            raise HTTPException(status_code=409, detail="watch already running; call /v1/watch/stop first")
        _ensure_cli_built()
        args = ["watch", "--file", str(Path(req.file).expanduser().resolve()), "--out-dir", req.out_dir]
        if req.from_start:
            args.append("--from-start")
        cmd = [_node_exe(), str(CLI_JS), *args]
        watch_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            env=os.environ.copy(),
            cwd=str(BRIDGE_ROOT),
        )
    return {"started": True, "cmd": cmd, "pid": watch_proc.pid}


@app.post("/v1/watch/stop")
async def watch_stop() -> dict[str, Any]:
    global watch_proc
    async with watch_lock:
        if watch_proc is None or watch_proc.poll() is not None:
            watch_proc = None
            return {"stopped": False, "message": "no active watch"}
        pid = watch_proc.pid
        watch_proc.terminate()
        try:
            watch_proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            watch_proc.kill()
            watch_proc.wait(timeout=3)
        watch_proc = None
    return {"stopped": True, "pid": pid}


@app.get("/v1/watch/status")
async def watch_status() -> dict[str, Any]:
    async with watch_lock:
        if watch_proc is None:
            return {"running": False}
        rc = watch_proc.poll()
        return {"running": rc is None, "pid": watch_proc.pid, "returncode": rc}
