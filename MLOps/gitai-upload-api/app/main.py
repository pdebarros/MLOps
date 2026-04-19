"""
Authenticated upload service:
- /auth/register and /auth/login backed by BigQuery users table
- /ingest/python requires valid user session token (and optional app API key)
- Stores uploaded Python additions as JSON blobs in GCS
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from google.cloud import bigquery, storage
from pydantic import BaseModel, EmailStr, Field

from app.config import settings

app = FastAPI(title="git-ai Python upload", version="2.0.0")

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


class AuthRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=256)


class AuthResponse(BaseModel):
    user_id: str
    email: EmailStr
    session_token: str
    expires_at: str
    created: bool = False


def _fetch_user(email: str) -> dict[str, Any] | None:
    client = _bq_client()
    sql = f"""
        SELECT user_id, email, password_hash, is_active, session_token_hash, session_expires_at
        FROM `{_users_table_ref()}`
        WHERE email = @email
        ORDER BY created_at DESC
        LIMIT 1
    """
    cfg = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email.lower())]
    )
    rows = list(client.query(sql, job_config=cfg).result())
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
    sql = f"""
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
        client.query(sql, job_config=cfg).result()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to insert user: {e}") from e
    return {"user_id": user_id, "email": email.lower()}


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


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "git-ai-upload-api", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/register", response_model=AuthResponse)
def register(body: AuthRequest, x_api_key: str | None = Security(api_key_header)) -> AuthResponse:
    _require_app_key(x_api_key)
    existing = _fetch_user(body.email.lower())
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")
    created = _insert_user(body.email.lower(), body.password)
    token, expires = _set_session(created["email"])
    return AuthResponse(
        user_id=created["user_id"],
        email=created["email"],
        session_token=token,
        expires_at=expires,
        created=True,
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
