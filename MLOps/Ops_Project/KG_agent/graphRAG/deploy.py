#!/usr/bin/env python3
"""
Deploy the graphRAG ADK agent to Google Cloud Agent Engine (Vertex AI Agent Runtime).

This wraps the official ADK CLI::

    adk deploy agent_engine --project=... --region=... <agent_folder>

Prerequisites
-------------
- gcloud installed; ``gcloud auth login`` and ``gcloud auth application-default login``
- Vertex AI / Agent Platform API enabled on the project
- ``adk`` on PATH (same venv you use for ``adk run KG_agent.graphRAG``)

Environment / secrets
---------------------
Neo4j, RAG settings (BigQuery ``rag_corpus_id`` lookup and/or corpus path env vars),
and model settings are read at runtime from environment variables (see ``config.py``).
Pass ``--env_file`` pointing at ``KG_agent/.env`` or configure variables on the
deployed Agent Engine resource in the console.

Usage examples
--------------
From this directory::

    python deploy.py --project my-gcp-project --region us-central1

With an env file and update-in-place::

    python deploy.py -p my-gcp-project -r us-central1 \\
        --env-file ../.env --agent-engine-id 751619551677906944
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _default_project() -> str | None:
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCP_PROJECT")
        or None
    )


def _default_region() -> str | None:
    return (
        os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("VERTEX_LOCATION")
        or None
    )


def _build_parser() -> argparse.ArgumentParser:
    here = Path(__file__).resolve().parent
    default_req = here / "requirements.txt"
    default_env = here.parent / ".env"

    p = argparse.ArgumentParser(
        description="Deploy graphRAG to Vertex AI Agent Engine via `adk deploy agent_engine`.",
    )
    p.add_argument(
        "-p",
        "--project",
        default=_default_project(),
        help="GCP project id (default: GOOGLE_CLOUD_PROJECT or GCP_PROJECT).",
    )
    p.add_argument(
        "-r",
        "--region",
        default=_default_region(),
        help="Agent Engine region (default: GOOGLE_CLOUD_LOCATION or VERTEX_LOCATION).",
    )
    p.add_argument(
        "--display-name",
        default="GraphRAG Experience Agent",
        help="Display name in the Agent Engine console.",
    )
    p.add_argument(
        "--description",
        default="",
        help="Optional description for the Agent Engine resource.",
    )
    p.add_argument(
        "--agent-engine-id",
        default=None,
        help="Existing reasoning engine id to update (numeric id, not full resource name).",
    )
    p.add_argument(
        "--agent-dir",
        type=Path,
        default=here,
        help=f"Agent source folder (default: {here}).",
    )
    p.add_argument(
        "--requirements-file",
        type=Path,
        default=default_req,
        help="requirements.txt for the remote runtime.",
    )
    p.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help=f"Optional .env to bundle (default: {default_env} if that file exists).",
    )
    p.add_argument(
        "--no-env-file",
        action="store_true",
        help="Do not pass --env_file; configure secrets on the Agent Engine resource instead.",
    )
    p.add_argument(
        "--trace-to-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Cloud Trace (default: true).",
    )
    p.add_argument(
        "--otel-to-cloud",
        action="store_true",
        help="Enable OpenTelemetry export to GCP.",
    )
    p.add_argument(
        "--validate-agent-import",
        action="store_true",
        help="Run ADK pre-deploy import check (needs local deps installed).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the adk command and exit without deploying.",
    )
    p.add_argument(
        "--keep-staging-dir",
        action="store_true",
        help="Keep the generated staging directory for inspection/debugging.",
    )
    return p


_LOCAL_ONLY_ENV_KEYS = {
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "PYTHONPATH",
    "VIRTUAL_ENV",
}

_AGENT_FILES = (
    "__init__.py",
    "agent.py",
    "config.py",
    "prompt.py",
    "tools.py",
    "requirements.txt",
)


def _looks_like_local_path(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    return (
        v.startswith("/Users/")
        or v.startswith("/private/")
        or v.startswith("~/")
    )


def _sanitize_env_file(src: Path, dst: Path) -> tuple[int, int]:
    kept = 0
    dropped = 0
    out: list[str] = []
    for raw in src.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in raw:
            out.append(raw)
            continue
        key, _, value = raw.partition("=")
        key = key.strip()
        v = value.strip()
        if key in _LOCAL_ONLY_ENV_KEYS:
            dropped += 1
            continue
        # Defensive: don't ship local absolute paths in *_FILE variables.
        if key.endswith("_FILE") and _looks_like_local_path(v):
            dropped += 1
            continue
        out.append(raw)
        kept += 1
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return kept, dropped


def _copy_required_tree(agent_dir: Path, staging_root: Path) -> Path:
    """
    Build a tiny deploy tree containing only the KG_agent package and graphRAG files.
    """
    kg_root = agent_dir.parent
    pkg_root = staging_root / "KG_agent"
    staged_agent = pkg_root / "graphRAG"
    staged_agent.mkdir(parents=True, exist_ok=True)

    # Keep package boundary so relative imports continue to resolve.
    shutil.copy2(kg_root / "__init__.py", pkg_root / "__init__.py")
    for name in _AGENT_FILES:
        shutil.copy2(agent_dir / name, staged_agent / name)
    return staged_agent


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    adk = shutil.which("adk")
    if not adk and not args.dry_run:
        print("error: `adk` not found on PATH. Install google-adk and activate your venv.", file=sys.stderr)
        return 1

    if not args.project:
        print("error: pass --project or set GOOGLE_CLOUD_PROJECT / GCP_PROJECT.", file=sys.stderr)
        return 1
    if not args.region:
        print(
            "error: pass --region or set GOOGLE_CLOUD_LOCATION / VERTEX_LOCATION.",
            file=sys.stderr,
        )
        return 1

    agent_dir = args.agent_dir.resolve()
    if not (agent_dir / "agent.py").is_file():
        print(f"error: no agent.py under {agent_dir}", file=sys.stderr)
        return 1

    req = args.requirements_file.resolve()
    if not req.is_file():
        print(f"error: requirements file not found: {req}", file=sys.stderr)
        return 1

    env_file: Path | None = None
    if args.no_env_file:
        env_file = None
    elif args.env_file is not None:
        env_file = args.env_file
    else:
        candidate = agent_dir.parent / ".env"
        if candidate.is_file():
            env_file = candidate
    if env_file is not None:
        env_file = env_file.resolve()
        if not env_file.is_file():
            print(f"error: --env-file not found: {env_file}", file=sys.stderr)
            return 1

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    staging_parent = Path(tempfile.gettempdir()) / f"graphRAG_deploy_{stamp}"
    staged_agent_dir = _copy_required_tree(agent_dir, staging_parent)

    sanitized_env_file: Path | None = None
    if env_file is not None:
        sanitized_env_file = staging_parent / "sanitized.env"
        kept, dropped = _sanitize_env_file(env_file, sanitized_env_file)
        print(
            f"Sanitized env file: kept {kept} vars, dropped {dropped} local-only vars."
        )

    cmd: list[str] = [
        adk or "adk",
        "deploy",
        "agent_engine",
        "--project",
        args.project,
        "--region",
        args.region,
        "--display_name",
        args.display_name,
        "--requirements_file",
        str(req),
    ]

    if args.description:
        cmd.extend(["--description", args.description])
    if args.agent_engine_id:
        cmd.extend(["--agent_engine_id", str(args.agent_engine_id)])
    if sanitized_env_file is not None:
        cmd.extend(["--env_file", str(sanitized_env_file)])
    if args.trace_to_cloud:
        cmd.append("--trace_to_cloud")
    else:
        cmd.append("--no-trace_to_cloud")
    if args.otel_to_cloud:
        cmd.append("--otel_to_cloud")
    if args.validate_agent_import:
        cmd.append("--validate-agent-import")

    cmd.append(str(staged_agent_dir))

    print("Running:", " ".join(cmd))
    print("Staging deploy source at:", staging_parent)
    if args.dry_run:
        if not args.keep_staging_dir:
            shutil.rmtree(staging_parent, ignore_errors=True)
        return 0

    try:
        proc = subprocess.run(cmd, check=False, cwd=staging_parent)
        return int(proc.returncode)
    finally:
        if args.keep_staging_dir:
            print("Keeping staging dir:", staging_parent)
        else:
            shutil.rmtree(staging_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
