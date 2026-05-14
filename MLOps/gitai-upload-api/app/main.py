"""
Authenticated upload service:
- /auth/register and /auth/login backed by BigQuery users table
- /ingest/python requires valid user session token (and optional app API key)
- Stores uploaded Python additions as JSON blobs in GCS
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from google.cloud import bigquery, storage
from pydantic import BaseModel, EmailStr, Field

from app.config import settings


logger = logging.getLogger("gitai_upload_api")

app = FastAPI(title="git-ai Python upload", version="2.0.0")

# Bucket for submitted Python source files — must match KG_agent's GCS_BUCKET_NAME
# so prod_pipeline can locate uploaded files. Configured via PYTHON_FILES_BUCKET or
# GCS_BUCKET_NAME env var; falls back to the KG agent default.
PYTHON_FILES_BUCKET = settings.python_files_bucket

_origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-User-Email",
        "X-Session-Token",
    ],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
user_email_header = APIKeyHeader(name="X-User-Email", auto_error=False)
session_token_header = APIKeyHeader(name="X-Session-Token", auto_error=False)

_kg_jobs_lock = threading.Lock()
_kg_jobs: dict[str, dict[str, Any]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_segment(path: str) -> str:
    s = path.replace("\\", "/").strip("/")
    s = re.sub(r"[^a-zA-Z0-9._/-]+", "_", s)
    return s[:180] if len(s) > 180 else s or "unknown"


def _users_table_ref() -> str:
    project = settings.bq_project_id or bigquery.Client().project
    return f"{project}.{settings.bq_dataset}.{settings.bq_users_table}"


def _bq_client() -> bigquery.Client:
    if settings.bq_project_id:
        return bigquery.Client(project=settings.bq_project_id)
    return bigquery.Client()


def _hash_password(password: str, *, salt: str | None = None) -> str:
    if not salt:
        salt = secrets.token_hex(16)
    iters = 200_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iters)
    return f"pbkdf2_sha256${iters}${salt}${dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        alg, iter_s, salt, hex_hash = stored.split("$", 3)
        if alg != "pbkdf2_sha256":
            return False
        iters = int(iter_s)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iters)
        return secrets.compare_digest(dk.hex(), hex_hash)
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _require_app_key(key: str | None) -> None:
    # Optional in dev; if API_KEY is set, enforce it
    if settings.api_key and key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


class AddedLine(BaseModel):
    line: int = Field(..., ge=1)
    text: str


class PythonIngest(BaseModel):
    workspace: str = ""
    file_path: str
    added_lines: list[AddedLine] = Field(default_factory=list)
    recorded_at: str = Field(default_factory=lambda: _now().isoformat())
    session_id: str | None = None


class PythonFileSubmission(BaseModel):
    workspace: str = ""
    file_path: str
    file_content: str
    recorded_at: str = Field(default_factory=lambda: _now().isoformat())
    session_id: str | None = None


class PythonFilesSubmission(BaseModel):
    workspace: str = ""
    files: list[PythonFileSubmission] = Field(default_factory=list)
    recorded_at: str = Field(default_factory=lambda: _now().isoformat())
    session_id: str | None = None


class AuthRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class AuthResponse(BaseModel):
    user_id: str
    email: EmailStr
    session_token: str
    expires_at: str
    created: bool = False
    rag_corpus_id: str | None = None


def _fetch_user(email: str) -> dict[str, Any] | None:
    """
    Fetch a user row by email. Tries to read `rag_corpus_id`; falls back to
    the legacy schema (without that column) so this service keeps working on
    pre-existing users tables until they're migrated.
    """
    client = _bq_client()
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email.lower())]
    )
    sql_with_corpus = f"""
        SELECT user_id, email, password_hash, is_active, session_token_hash,
               session_expires_at, rag_corpus_id
        FROM `{_users_table_ref()}`
        WHERE email = @email
        ORDER BY created_at DESC
        LIMIT 1
    """
    try:
        rows = list(client.query(sql_with_corpus, job_config=cfg).result())
        has_corpus_col = True
    except Exception:
        sql_legacy = f"""
            SELECT user_id, email, password_hash, is_active, session_token_hash,
                   session_expires_at
            FROM `{_users_table_ref()}`
            WHERE email = @email
            ORDER BY created_at DESC
            LIMIT 1
        """
        rows = list(client.query(sql_legacy, job_config=cfg).result())
        has_corpus_col = False
    if not rows:
        return None
    r = rows[0]
    return {
        "user_id": r["user_id"],
        "email": r["email"],
        "password_hash": r["password_hash"],
        "is_active": bool(r["is_active"]),
        "session_token_hash": r["session_token_hash"],
        "session_expires_at": r["session_expires_at"],
        "rag_corpus_id": r["rag_corpus_id"] if has_corpus_col else None,
    }


def _insert_user(email: str, password: str) -> dict[str, str]:
    """
    Use standard SQL INSERT (query job), not insert_rows_json streaming API.
    Streaming inserts place rows in the streaming buffer; BigQuery forbids UPDATE/DELETE
    on those rows until they commit, which breaks immediate session updates.
    """
    user_id = str(uuid.uuid4())
    password_hash = _hash_password(password)
    client = _bq_client()
    sql_with_corpus = f"""
        INSERT INTO `{_users_table_ref()}` (
            user_id,
            email,
            password_hash,
            created_at,
            last_login_at,
            is_active,
            session_token_hash,
            session_expires_at,
            total_upload_requests,
            total_uploaded_lines,
            rag_corpus_id
        )
        VALUES (
            @user_id,
            @email,
            @password_hash,
            CURRENT_TIMESTAMP(),
            NULL,
            TRUE,
            NULL,
            NULL,
            0,
            0,
            NULL
        )
    """
    sql_legacy = f"""
        INSERT INTO `{_users_table_ref()}` (
            user_id,
            email,
            password_hash,
            created_at,
            last_login_at,
            is_active,
            session_token_hash,
            session_expires_at,
            total_upload_requests,
            total_uploaded_lines
        )
        VALUES (
            @user_id,
            @email,
            @password_hash,
            CURRENT_TIMESTAMP(),
            NULL,
            TRUE,
            NULL,
            NULL,
            0,
            0
        )
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("email", "STRING", email.lower()),
            bigquery.ScalarQueryParameter("password_hash", "STRING", password_hash),
        ]
    )
    try:
        client.query(sql_with_corpus, job_config=cfg).result()
    except Exception:
        try:
            client.query(sql_legacy, job_config=cfg).result()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to insert user: {e}") from e
    return {"user_id": user_id, "email": email.lower()}


def _set_user_rag_corpus(email: str, rag_corpus_id: str) -> None:
    """Persist the Vertex RAG corpus numeric id on the user row."""
    client = _bq_client()
    sql = f"""
        UPDATE `{_users_table_ref()}`
        SET rag_corpus_id = @rag_corpus_id
        WHERE email = @email
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("rag_corpus_id", "STRING", rag_corpus_id),
            bigquery.ScalarQueryParameter("email", "STRING", email.lower()),
        ]
    )
    client.query(sql, job_config=cfg).result()


def _rag_corpus_display_name(user_id: str) -> str:
    # Vertex RAG display names accept up to 128 chars and a constrained charset.
    # User IDs are uuid4 hex, so this is always safe.
    return f"gitai-{user_id}"


def _extract_rag_corpus_id(resource_name: str) -> str:
    """
    Extract the numeric id (last path segment) from a Vertex RAG corpus
    resource name like ``projects/.../locations/.../ragCorpora/<id>``.

    Falls back to the original string if it doesn't match the expected shape
    so downstream code never gets an empty value.
    """
    if not resource_name:
        return ""
    tail = resource_name.rstrip("/").rsplit("/", 1)[-1]
    return tail or resource_name


def _create_user_rag_corpus(user_id: str, email: str) -> str:
    """
    Create an empty Vertex AI RAG Engine corpus for a freshly registered user.

    Uses ``EmbeddingModelConfig`` with a publisher embedding model (default
    ``text-embedding-004``), which provisions a **Spanner-backed** corpus in
    the configured region (default ``europe-west4``), not Serverless Vector Search.

    Returns the corpus **numeric id** (the last segment of the Vertex resource
    name). Reconstruct the full resource name as
    ``projects/{VERTEX_PROJECT}/locations/{VERTEX_LOCATION}/ragCorpora/{id}``
    when calling the SDK. Raises on failure so the caller can decide whether
    to fail the registration.
    """
    if not settings.vertex_project:
        raise RuntimeError(
            "Vertex project is not configured: set VERTEX_PROJECT or BQ_PROJECT_ID."
        )

    import vertexai  # noqa: PLC0415 — lazy import; only needed during register
    from vertexai import rag  # noqa: PLC0415

    vertexai.init(project=settings.vertex_project, location=settings.vertex_location)

    display_name = _rag_corpus_display_name(user_id)
    description = f"gitai per-user RAG corpus for {email}"

    emb_config = rag.EmbeddingModelConfig(
        publisher_model=settings.rag_embedding_publisher_model
    )

    try:
        corpus = rag.create_corpus(
            display_name=display_name,
            description=description,
            embedding_model_config=emb_config,
        )
    except TypeError:
        # Older SDKs may not accept `description`.
        corpus = rag.create_corpus(
            display_name=display_name,
            embedding_model_config=emb_config,
        )

    name = getattr(corpus, "name", "") or ""
    if not name:
        raise RuntimeError("Vertex returned a corpus without a resource name.")
    corpus_id = _extract_rag_corpus_id(name)
    if not corpus_id:
        raise RuntimeError(f"Could not parse corpus id from resource name: {name!r}")
    return corpus_id


def _set_session(email: str) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    token_hash = _token_hash(token)
    expires = _now() + timedelta(hours=settings.session_ttl_hours)
    client = _bq_client()
    sql = f"""
        UPDATE `{_users_table_ref()}`
        SET
          session_token_hash = @session_token_hash,
          session_expires_at = @session_expires_at,
          last_login_at = @last_login_at
        WHERE email = @email
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_token_hash", "STRING", token_hash),
            bigquery.ScalarQueryParameter("session_expires_at", "TIMESTAMP", expires),
            bigquery.ScalarQueryParameter("last_login_at", "TIMESTAMP", _now()),
            bigquery.ScalarQueryParameter("email", "STRING", email.lower()),
        ]
    )
    client.query(sql, job_config=cfg).result()
    return token, expires.isoformat()


def _authenticate_user_session(email: str | None, token: str | None) -> dict[str, Any]:
    if not email or not token:
        raise HTTPException(status_code=401, detail="Missing user auth headers")
    user = _fetch_user(email.lower())
    if not user or not user["is_active"]:
        raise HTTPException(status_code=401, detail="Unknown or inactive user")
    if not user.get("session_token_hash"):
        raise HTTPException(status_code=401, detail="No active session; log in again")
    if user.get("session_expires_at") and user["session_expires_at"] < _now():
        raise HTTPException(status_code=401, detail="Session expired; log in again")
    if not secrets.compare_digest(_token_hash(token), user["session_token_hash"]):
        raise HTTPException(status_code=401, detail="Invalid session token")
    return user


def _increment_usage(email: str, lines_count: int) -> None:
    client = _bq_client()
    sql = f"""
        UPDATE `{_users_table_ref()}`
        SET
          total_upload_requests = COALESCE(total_upload_requests, 0) + 1,
          total_uploaded_lines = COALESCE(total_uploaded_lines, 0) + @line_count
        WHERE email = @email
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("line_count", "INT64", lines_count),
            bigquery.ScalarQueryParameter("email", "STRING", email.lower()),
        ]
    )
    client.query(sql, job_config=cfg).result()


def _upload_to_gcs(payload: PythonIngest, user_id: str) -> str:
    if not settings.gcs_bucket:
        raise HTTPException(status_code=503, detail="Server misconfigured: GCS_BUCKET not set")
    client = storage.Client()
    bucket = client.bucket(settings.gcs_bucket)
    day = _now().strftime("%Y-%m-%d")
    oid = uuid.uuid4().hex[:12]
    rel = _safe_segment(payload.file_path)
    # user-scoped object path helps monitor usage and isolate data
    name = f"{settings.gcs_object_prefix}/{user_id}/{day}/{rel}_{oid}.json"
    blob = bucket.blob(name)
    body = json.dumps(payload.model_dump(), ensure_ascii=False, indent=2)
    blob.upload_from_string(body.encode("utf-8"), content_type="application/json; charset=utf-8")
    return f"gs://{settings.gcs_bucket}/{name}"


def _folder_marker_name(user_id: str) -> str:
    return f"{user_id}/python_files/"


def _python_object_name(user_id: str, file_path: str) -> str:
    rel = _safe_segment(file_path)
    if not rel.endswith(".py"):
        rel = f"{rel}.py"
    return f"{user_id}/python_files/{rel}"


def _store_python_file_to_gcs(payload: PythonFileSubmission, user_id: str) -> tuple[str, str, bool]:
    """
    Store submitted Python source in a fixed bucket under <user_id>/python_files/.
    Returns (gcs_uri, status, already_existed).
    """
    client = storage.Client()
    bucket = client.bucket(PYTHON_FILES_BUCKET)

    # Optional marker object so the prefix is visible as a "folder" in some UIs.
    marker_name = _folder_marker_name(user_id)
    marker_blob = bucket.blob(marker_name)
    if not marker_blob.exists():
        marker_blob.upload_from_string(b"", content_type="application/x-directory")

    name = _python_object_name(user_id, payload.file_path)
    blob = bucket.blob(name)
    existed = blob.exists()
    existing_text = blob.download_as_text(encoding="utf-8") if existed else None

    if existed and (existing_text or "") == payload.file_content:
        return f"gs://{PYTHON_FILES_BUCKET}/{name}", "already_exists_same_content", True

    blob.upload_from_string(payload.file_content.encode("utf-8"), content_type="text/x-python; charset=utf-8")
    status = "updated_existing_file" if existed else "stored_new_file"
    return f"gs://{PYTHON_FILES_BUCKET}/{name}", status, existed


def _prod_pipeline_script_path() -> Path:
    if settings.kg_pipeline_script:
        return Path(settings.kg_pipeline_script)
    root = Path(__file__).resolve().parents[2]
    return root / "Ops_Project" / "KG_agent" / "prod_pipeline.py"


def _kg_pipeline_python() -> str:
    return settings.kg_pipeline_python or sys.executable


# ── Cloud Run Jobs helpers ────────────────────────────────────────────────────

def _trigger_cloud_run_job(user_id: str) -> str:
    """
    Create a Cloud Run Job execution for prod_pipeline.py.

    Returns the Cloud Run execution resource name which serves as the
    stable, instance-restart-safe job ID for status polling.
    """
    from google.cloud import run_v2  # noqa: PLC0415 – lazy import, only on Cloud Run

    client = run_v2.JobsClient()
    job_resource = (
        f"projects/{settings.cloud_run_project}"
        f"/locations/{settings.cloud_run_region}"
        f"/jobs/{settings.cloud_run_kg_job_name}"
    )

    # Pass --user-id as CMD args; static secrets/config live in the Job definition.
    args_override = ["--user-id", user_id]
    if settings.kg_pipeline_database:
        args_override += ["--neo4j-database", settings.kg_pipeline_database]

    # Forward optional hyperparameter overrides as env vars (read by prod_pipeline defaults).
    from google.cloud.run_v2.types import EnvVar  # noqa: PLC0415

    env_overrides: list[EnvVar] = []
    _hp: list[tuple[str, int | None]] = [
        ("PIPELINE_STRUCTURAL_KG_BATCH",    settings.kg_structural_batch_size),
        ("PIPELINE_STRUCTURAL_KG_OVERLAP",  settings.kg_structural_batch_overlap),
        ("PIPELINE_STRUCTURAL_MAX_DOC_CHARS", settings.kg_structural_max_doc_chars),
        ("PIPELINE_TECHNICAL_KG_BATCH",     settings.kg_technical_batch_size),
        ("PIPELINE_TECHNICAL_KG_OVERLAP",   settings.kg_technical_batch_overlap),
        ("PIPELINE_TECHNICAL_MAX_DOC_CHARS", settings.kg_technical_max_doc_chars),
    ]
    for env_name, val in _hp:
        if val is not None:
            env_overrides.append(EnvVar(name=env_name, value=str(val)))

    request = run_v2.RunJobRequest(
        name=job_resource,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    args=args_override,
                    env=env_overrides,
                )
            ],
            task_count=1,
        ),
    )

    operation = client.run_job(request=request)
    # operation.metadata is the Execution proto; .name is the resource path.
    execution_name: str = operation.metadata.name
    return execution_name


def _get_cloud_run_execution_status(execution_name: str) -> dict[str, Any]:
    """Query Cloud Run for the current state of a job execution."""
    from google.cloud import run_v2  # noqa: PLC0415

    client = run_v2.ExecutionsClient()
    execution = client.get_execution(name=execution_name)

    if execution.completion_time:
        finished_at = execution.completion_time.isoformat()
        status = "success" if execution.failed_count == 0 else "error"
        return {"status": status, "finished_at": finished_at}

    return {"status": "running"}


def _run_prod_pipeline(user_id: str) -> dict[str, Any]:
    """
    Trigger KG build after upload.
    Uses a subprocess so this API stays decoupled from KG_agent imports/runtime state.

    All tunable hyperparameters are forwarded from settings (env vars), so the
    structural and technical KG tracks can be configured without code changes.
    """
    script = _prod_pipeline_script_path()
    if not script.is_file():
        return {
            "status": "error",
            "reason": "prod_pipeline_not_found",
            "path": str(script),
        }

    cmd = [_kg_pipeline_python(), str(script), "--user-id", user_id]
    #cmd.extend(["--neo4j-database", settings.kg_pipeline_database or "neo4j"])

    # Forward optional hyperparameter overrides (structural track)
    if settings.kg_structural_batch_size is not None:
        cmd.extend(["--structural-batch-size", str(settings.kg_structural_batch_size)])
    if settings.kg_structural_batch_overlap is not None:
        cmd.extend(["--structural-batch-overlap", str(settings.kg_structural_batch_overlap)])
    if settings.kg_structural_max_doc_chars is not None:
        cmd.extend(["--structural-max-doc-chars", str(settings.kg_structural_max_doc_chars)])

    # Forward optional hyperparameter overrides (technical track)
    if settings.kg_technical_batch_size is not None:
        cmd.extend(["--technical-batch-size", str(settings.kg_technical_batch_size)])
    if settings.kg_technical_batch_overlap is not None:
        cmd.extend(["--technical-batch-overlap", str(settings.kg_technical_batch_overlap)])
    if settings.kg_technical_max_doc_chars is not None:
        cmd.extend(["--technical-max-doc-chars", str(settings.kg_technical_max_doc_chars)])

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            check=False,
            timeout=settings.kg_pipeline_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "exit_code": None,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
        }
    except Exception as exc:
        return {
            "status": "error",
            "reason": f"pipeline_launch_failed: {exc}",
        }

    return {
        "status": "success" if proc.returncode == 0 else "error",
        "exit_code": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-4000:],
        "stderr_tail": (proc.stderr or "")[-4000:],
    }


def _kg_job_view(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "user_id": job["user_id"],
        "status": job["status"],
        "created_at": job["created_at"],
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "result": job.get("result"),
    }


def _worker_headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if settings.kg_worker_api_key:
        headers["X-Worker-Api-Key"] = settings.kg_worker_api_key
    return headers


def _worker_request_json(path: str, *, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if not settings.kg_worker_base_url:
        raise RuntimeError("KG_WORKER_BASE_URL is not configured")
    url = f"{settings.kg_worker_base_url}{path}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method=method,
        headers=_worker_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"worker HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"worker unreachable: {e}") from e


def _run_kg_job(job_id: str) -> None:
    with _kg_jobs_lock:
        job = _kg_jobs.get(job_id)
        if not job:
            return
        job["status"] = "running"
        job["started_at"] = _now().isoformat()
    result = _run_prod_pipeline(job["user_id"])
    with _kg_jobs_lock:
        job2 = _kg_jobs.get(job_id)
        if not job2:
            return
        job2["result"] = result
        job2["finished_at"] = _now().isoformat()
        job2["status"] = "success" if result.get("status") == "success" else "error"


def _enqueue_kg_job(user_id: str) -> str:
    job_id = uuid.uuid4().hex
    job: dict[str, Any] = {
        "job_id": job_id,
        "user_id": user_id,
        "status": "queued",
        "created_at": _now().isoformat(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        # execution_name is set when using Cloud Run Jobs; used for durable status polling.
        "execution_name": None,
        # remote_job_id is set when using a KG worker service.
        "remote_job_id": None,
    }
    with _kg_jobs_lock:
        _kg_jobs[job_id] = job

    if settings.cloud_run_kg_job_name:
        # ── Cloud Run Jobs mode ──────────────────────────────────────────
        # Durable: execution status survives Cloud Run instance restarts.
        try:
            execution_name = _trigger_cloud_run_job(user_id)
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job2["execution_name"] = execution_name
                    job2["status"] = "running"
                    job2["started_at"] = _now().isoformat()
        except Exception as exc:
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job2["status"] = "error"
                    job2["finished_at"] = _now().isoformat()
                    job2["result"] = {"status": "error", "reason": f"cloud_run_job_trigger_failed: {exc}"}
    elif settings.kg_worker_base_url:
        # ── Remote KG worker service mode ───────────────────────────────
        try:
            created = _worker_request_json(
                "/jobs",
                method="POST",
                payload={"user_id": user_id, "neo4j_database": settings.kg_pipeline_database or None},
            )
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job2["remote_job_id"] = str(created.get("job_id", ""))
                    job2["status"] = str(created.get("status") or "queued")
        except Exception as exc:
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job2["status"] = "error"
                    job2["finished_at"] = _now().isoformat()
                    job2["result"] = {"status": "error", "reason": f"remote_enqueue_failed: {exc}"}
    else:
        # ── Local background thread mode (dev/local only) ────────────────
        t = threading.Thread(target=_run_kg_job, args=(job_id,), daemon=True)
        t.start()

    return job_id


def _normalize_python_files_payload(
    body: PythonFileSubmission | PythonFilesSubmission,
) -> list[PythonFileSubmission]:
    if isinstance(body, PythonFileSubmission):
        return [body]
    return list(body.files or [])


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "git-ai-upload-api", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/kg/jobs/{job_id}")
def get_kg_job_status(
    job_id: str,
    x_api_key: str | None = Security(api_key_header),
    x_user_email: str | None = Security(user_email_header),
    x_session_token: str | None = Security(session_token_header),
) -> dict[str, Any]:
    _require_app_key(x_api_key)
    user = _authenticate_user_session(x_user_email, x_session_token)
    with _kg_jobs_lock:
        job = _kg_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="KG job not found")
        if job["user_id"] != user["user_id"]:
            raise HTTPException(status_code=403, detail="Forbidden for this job")
        execution_name = str(job.get("execution_name") or "")
        remote_job_id = str(job.get("remote_job_id") or "")

    if execution_name:
        # ── Cloud Run Jobs: query execution status directly ──────────────
        # Durable: works even after instance restart because we query Cloud Run.
        try:
            cr = _get_cloud_run_execution_status(execution_name)
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job2["status"] = cr["status"]
                    if cr.get("finished_at"):
                        job2["finished_at"] = cr["finished_at"]
                    job = job2
        except Exception as exc:
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job = job2
            # Non-fatal: return last known in-memory status if Cloud Run API call fails.
            _ = exc
    elif remote_job_id:
        # ── Remote KG worker service ─────────────────────────────────────
        try:
            remote = _worker_request_json(f"/jobs/{remote_job_id}", method="GET")
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job2["status"] = str(remote.get("status") or job2["status"])
                    job2["started_at"] = remote.get("started_at")
                    job2["finished_at"] = remote.get("finished_at")
                    job2["result"] = remote.get("result")
                    job = job2
        except Exception as exc:
            with _kg_jobs_lock:
                job2 = _kg_jobs.get(job_id)
                if job2:
                    job2["status"] = "error"
                    job2["result"] = {"status": "error", "reason": f"remote_status_failed: {exc}"}
                    job = job2
    return _kg_job_view(job)


@app.post("/auth/register", response_model=AuthResponse)
def register(body: AuthRequest, x_api_key: str | None = Security(api_key_header)) -> AuthResponse:
    _require_app_key(x_api_key)
    existing = _fetch_user(body.email.lower())
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    created = _insert_user(body.email.lower(), body.password)

    # Create an empty Vertex AI RAG Engine corpus for the new user. This is
    # the user's private knowledge base; later uploads / KG runs can populate
    # it. Failures are non-fatal by default so a Vertex outage doesn't block
    # registration; flip RAG_CORPUS_REQUIRED_ON_REGISTER=1 to make it strict.
    rag_corpus_id: str | None = None
    try:
        rag_corpus_id = _create_user_rag_corpus(created["user_id"], created["email"])
    except Exception as exc:
        logger.exception("Failed to create RAG corpus for user %s", created["user_id"])
        if settings.rag_corpus_required_on_register:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to create RAG corpus: {exc}",
            ) from exc

    if rag_corpus_id:
        try:
            _set_user_rag_corpus(created["email"], rag_corpus_id)
        except Exception:
            # The corpus exists in Vertex; we just couldn't persist its id.
            # Surface this in logs but don't fail registration.
            logger.exception(
                "Created RAG corpus %s but failed to persist on user row %s",
                rag_corpus_id,
                created["user_id"],
            )

    token, expires = _set_session(created["email"])
    return AuthResponse(
        user_id=created["user_id"],
        email=created["email"],
        session_token=token,
        expires_at=expires,
        created=True,
        rag_corpus_id=rag_corpus_id,
    )


@app.post("/auth/login", response_model=AuthResponse)
def login(body: AuthRequest, x_api_key: str | None = Security(api_key_header)) -> AuthResponse:
    _require_app_key(x_api_key)
    user = _fetch_user(body.email.lower())
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not user["is_active"]:
        raise HTTPException(status_code=403, detail="Account disabled")
    if not _verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect password")
    token, expires = _set_session(user["email"])
    return AuthResponse(
        user_id=user["user_id"],
        email=user["email"],
        session_token=token,
        expires_at=expires,
        created=False,
        rag_corpus_id=user.get("rag_corpus_id"),
    )


@app.post("/ingest/python")
def ingest_python(
    body: PythonIngest,
    x_api_key: str | None = Security(api_key_header),
    x_user_email: str | None = Security(user_email_header),
    x_session_token: str | None = Security(session_token_header),
) -> dict[str, str]:
    _require_app_key(x_api_key)
    user = _authenticate_user_session(x_user_email, x_session_token)
    if not body.added_lines:
        raise HTTPException(status_code=400, detail="added_lines must be non-empty")
    uri = _upload_to_gcs(body, user["user_id"])
    _increment_usage(user["email"], len(body.added_lines))
    return {"gcs_uri": uri, "status": "stored", "user_id": user["user_id"]}


@app.post("/ingest/python-file")
def ingest_python_file(
    body: PythonFileSubmission | PythonFilesSubmission,
    x_api_key: str | None = Security(api_key_header),
    x_user_email: str | None = Security(user_email_header),
    x_session_token: str | None = Security(session_token_header),
) -> dict[str, Any]:
    _require_app_key(x_api_key)
    user = _authenticate_user_session(x_user_email, x_session_token)
    files = _normalize_python_files_payload(body)
    if not files:
        raise HTTPException(status_code=400, detail="files must be non-empty")

    stored: list[dict[str, str]] = []
    total_lines = 0
    for item in files:
        if not item.file_path.strip():
            raise HTTPException(status_code=400, detail="file_path is required")
        if not item.file_content.strip():
            raise HTTPException(status_code=400, detail="file_content must be non-empty")
        uri, store_status, existed = _store_python_file_to_gcs(item, user["user_id"])
        line_count = len(item.file_content.splitlines())
        total_lines += line_count
        stored.append(
            {
                "file_path": item.file_path,
                "gcs_uri": uri,
                "status": store_status,
                "already_existed": "true" if existed else "false",
                "line_count": str(line_count),
            }
        )

    _increment_usage(user["email"], total_lines)
    job_id = _enqueue_kg_job(user["user_id"])

    return {
        "status": "stored_and_kg_queued",
        "user_id": user["user_id"],
        "bucket": PYTHON_FILES_BUCKET,
        "files_received": len(files),
        "total_uploaded_lines": total_lines,
        "files": stored,
        "kg_job_id": job_id,
        "kg_status_endpoint": f"/kg/jobs/{job_id}",
    }


