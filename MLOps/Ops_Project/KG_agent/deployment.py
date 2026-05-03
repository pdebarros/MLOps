"""
Deploy the KG ADK ``root_agent`` (``agent.py``) to Vertex AI Agent Engine.

Uses the same flow as the official ADK + Agent Engine quickstart: wrap the agent in
``AdkApp``, call ``vertexai.Client(...).agent_engines.create``, stage artifacts in GCS.

**Prerequisites**

- APIs: Vertex AI, Cloud Storage; billing enabled.
- IAM: typically ``roles/aiplatform.user`` and storage access on the staging bucket.
- Auth: ``gcloud auth application-default login``

**Install** (Agent Engine extras need a recent SDK)::

    pip install 'google-cloud-aiplatform[agent_engines,adk]>=1.112'

**Environment**

- ``GOOGLE_CLOUD_PROJECT`` (or ``GCP_PROJECT``) — required
- ``VERTEX_LOCATION`` or ``GOOGLE_CLOUD_REGION`` — optional (default from ``config``)
- ``VERTEX_AGENT_STAGING_BUCKET`` — GCS URI ``gs://...`` for staging (required for deploy)

Optional: Neo4j / GCS settings are forwarded to the remote runtime if set locally and you
do not pass ``--no-env-vars``.

**Run** (from ``Ops_Project``)::

    python KG_agent/deployment.py deploy --display-name kg-agent
    python KG_agent/deployment.py local-test --message "Hello"
    python KG_agent/deployment.py delete \\
        --name projects/PROJECT/locations/REGION/reasoningEngines/ENGINE_ID

See also: https://cloud.google.com/agent-builder/agent-engine/quickstart-adk
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

_KG = Path(__file__).resolve().parent
_OPS = _KG.parent
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kg_agent.deployment")

_ENV_KEYS_OPTIONAL = (
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "GCS_BUCKET_NAME",
    "GCS_PREFIX",
    "GOOGLE_CLOUD_PROJECT",
    "GCP_PROJECT",
    "VERTEX_LOCATION",
    "GOOGLE_CLOUD_REGION",
    "GEMINI_MODEL",
    "GEMINI_GRAPH_MODEL",
    "VERTEX_GEMINI_MODEL",
)


def _staging_bucket() -> str:
    raw = (os.environ.get("VERTEX_AGENT_STAGING_BUCKET") or "").strip()
    if not raw:
        raise SystemExit(
            "Set VERTEX_AGENT_STAGING_BUCKET to a GCS bucket, e.g. gs://my-staging-bucket"
        )
    if raw.startswith("gs://"):
        return raw
    return f"gs://{raw}"


def _env_vars_for_remote() -> dict[str, str] | None:
    out = {k: os.environ[k] for k in _ENV_KEYS_OPTIONAL if os.environ.get(k)}
    return out or None


def _ensure_vertex_sdk() -> None:
    try:
        import vertexai  # noqa: F401
        from vertexai import agent_engines  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "Missing Vertex Agent Engine SDK. Install with:\n"
            "  pip install 'google-cloud-aiplatform[agent_engines,adk]>=1.112'\n"
            f"Original error: {e}"
        ) from e


def _client():
    import vertexai

    from config import Config, google_cloud_project

    project = google_cloud_project()
    if not project:
        raise SystemExit("Set GOOGLE_CLOUD_PROJECT (or GCP_PROJECT) for deployment.")

    location = (Config.VERTEX_LOCATION or "us-central1").strip()
    if not hasattr(vertexai, "Client"):
        raise SystemExit(
            "vertexai.Client not found. Upgrade: pip install -U "
            "'google-cloud-aiplatform[agent_engines,adk]>=1.112'"
        )
    return vertexai.Client(project=project, location=location), project, location


def cmd_deploy(args: argparse.Namespace) -> None:
    _ensure_vertex_sdk()
    from vertexai import agent_engines

    from agent import root_agent

    client, project, location = _client()
    staging = _staging_bucket()
    req_file = _OPS / "requirements.txt"
    if not req_file.is_file():
        raise SystemExit(f"requirements.txt not found at {req_file}")

    env_vars = None if args.no_env_vars else _env_vars_for_remote()

    app = agent_engines.AdkApp(agent=root_agent)
    config: dict = {
        "requirements": str(req_file),
        "staging_bucket": staging,
        "extra_packages": [str(_KG)],
        "display_name": args.display_name,
        "description": args.description or "KG_agent ADK root_agent (Ops_Project)",
        "agent_framework": "google-adk",
    }
    if env_vars:
        config["env_vars"] = env_vars
    if args.gcs_dir_name:
        config["gcs_dir_name"] = args.gcs_dir_name

    logger.info(
        "Deploying KG_agent to Agent Engine (project=%s location=%s staging=%s)...",
        project,
        location,
        staging,
    )
    remote = client.agent_engines.create(agent=app, config=config)
    logger.info("Deployed. Resource: %s", getattr(remote, "api_resource", remote))
    print(getattr(remote, "api_resource", remote))


def cmd_delete(args: argparse.Namespace) -> None:
    _ensure_vertex_sdk()
    client, _, _ = _client()
    name = args.name.strip()
    if not name:
        raise SystemExit("--name is required (full reasoningEngines resource name).")
    logger.info("Deleting %s ...", name)
    client.agent_engines.delete(name=name, force=args.force)
    logger.info("Delete request submitted.")


async def _run_local_test(message: str, user_id: str) -> None:
    from vertexai import agent_engines

    from agent import root_agent

    app = agent_engines.AdkApp(agent=root_agent)
    async for event in app.async_stream_query(user_id=user_id, message=message):
        print(event)


def cmd_local_test(args: argparse.Namespace) -> None:
    _ensure_vertex_sdk()
    asyncio.run(_run_local_test(args.message, args.user_id))


def main() -> None:
    p = argparse.ArgumentParser(description="Deploy KG_agent ADK root_agent to Vertex AI Agent Engine.")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("deploy", help="Create a remote Agent Engine from root_agent.")
    d.add_argument(
        "--display-name",
        default="kg-agent",
        help="Display name in Vertex AI console.",
    )
    d.add_argument("--description", default=None, help="Optional description.")
    d.add_argument(
        "--gcs-dir-name",
        default=None,
        help="Subfolder under staging bucket for this build (avoids overwrites).",
    )
    d.add_argument(
        "--no-env-vars",
        action="store_true",
        help="Do not pass local Neo4j/GCS env vars to the remote agent.",
    )
    d.set_defaults(func=cmd_deploy)

    lt = sub.add_parser("local-test", help="Run AdkApp.async_stream_query locally (no deploy).")
    lt.add_argument("--message", default="Summarize the KG pipeline role.", help="User message.")
    lt.add_argument("--user-id", default="local-kg-test", help="ADK user_id (<=128 chars).")
    lt.set_defaults(func=cmd_local_test)

    rm = sub.add_parser("delete", help="Delete a reasoning engine by full resource name.")
    rm.add_argument(
        "--name",
        required=True,
        help="projects/PROJECT/locations/LOCATION/reasoningEngines/ID",
    )
    rm.add_argument(
        "--force",
        action="store_true",
        help="Also delete child resources (sessions, etc.).",
    )
    rm.set_defaults(func=cmd_delete)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
