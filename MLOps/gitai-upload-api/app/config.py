import os
from pathlib import Path


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


_load_dotenv()

# Helps ADC pick a quota/billing project for client libraries (reduces user-creds warning).
_bq_proj = os.environ.get("BQ_PROJECT_ID", "").strip()
if _bq_proj:
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", _bq_proj)


def _opt_int(key: str) -> int | None:
    raw = os.environ.get(key, "").strip()
    return int(raw) if raw else None


class Settings:
    """Populated from environment (Cloud Run) or a local .env file (via python-dotenv)."""

    def __init__(self) -> None:
        # ── Auth / upload ────────────────────────────────────────────────
        self.api_key = os.environ.get("API_KEY", "").strip()
        self.gcs_bucket = os.environ.get("GCS_BUCKET", "").strip()
        self.gcs_object_prefix = (os.environ.get("GCS_OBJECT_PREFIX") or "gitai-python").strip().rstrip("/") or "gitai-python"
        self.bq_project_id = os.environ.get("BQ_PROJECT_ID", "").strip()
        self.bq_dataset = os.environ.get("BQ_DATASET", "gitai").strip()
        self.bq_users_table = os.environ.get("BQ_USERS_TABLE", "users").strip()
        self.session_ttl_hours = int(os.environ.get("SESSION_TTL_HOURS", "24"))
        # Comma-separated list of allowed origins for browser/webview clients.
        # Example: "http://localhost:1420,tauri://localhost,https://app.example.com"
        self.cors_allow_origins = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost:1420,tauri://localhost").strip()

        # ── Python source file storage ───────────────────────────────────
        # Must match KG_agent's GCS_BUCKET_NAME so prod_pipeline can find uploaded files.
        self.python_files_bucket: str = (
            os.environ.get("PYTHON_FILES_BUCKET", "").strip()
            or os.environ.get("GCS_BUCKET_NAME", "").strip()
            or "codebases-04-03-26"
        )

        # ── KG pipeline — subprocess runner ─────────────────────────────
        # Absolute path to prod_pipeline.py; auto-detected relative to this repo if blank.
        self.kg_pipeline_script: str = os.environ.get("KG_PIPELINE_SCRIPT", "").strip()
        # Python interpreter that has KG_agent deps installed; defaults to sys.executable.
        self.kg_pipeline_python: str = os.environ.get("KG_PIPELINE_PYTHON", "").strip()
        # Neo4j logical database name. Leave blank to let prod_pipeline use its own default
        # (correct for AuraDB where the default database name varies per instance).
        self.kg_pipeline_database: str = os.environ.get("KG_PIPELINE_DATABASE", "").strip()
        # Max seconds to wait for the pipeline subprocess before giving up.
        self.kg_pipeline_timeout_seconds: int = int(os.environ.get("KG_PIPELINE_TIMEOUT_SECONDS", "600"))

        # ── Cloud Run Job for KG pipeline ────────────────────────────────
        # GCP project for Cloud Run Jobs; falls back to the BigQuery project.
        self.cloud_run_project: str = (
            os.environ.get("CLOUD_RUN_PROJECT", "").strip() or self.bq_project_id
        )
        # Region where the Cloud Run Job is deployed.
        self.cloud_run_region: str = (
            os.environ.get("CLOUD_RUN_REGION", "us-central1").strip() or "us-central1"
        )
        # Name of the pre-created Cloud Run Job that runs prod_pipeline.py.
        # When set, file uploads trigger a Cloud Run Job execution instead of a local subprocess.
        self.cloud_run_kg_job_name: str = os.environ.get("CLOUD_RUN_KG_JOB_NAME", "").strip()

        # ── KG pipeline — optional remote worker ─────────────────────────
        # When set, jobs are delegated to a separate KG worker service instead of running
        # in a background thread. Leave blank to run the pipeline in-process (default).
        self.kg_worker_base_url: str = os.environ.get("KG_WORKER_BASE_URL", "").strip()
        self.kg_worker_api_key: str = os.environ.get("KG_WORKER_API_KEY", "").strip()

        # ── KG pipeline — structural track hyperparameters ───────────────
        # Number of file summaries per KG extraction batch (larger = more cross-file context).
        self.kg_structural_batch_size: int | None = _opt_int("KG_STRUCTURAL_BATCH_SIZE")
        # Documents shared between adjacent structural batches (overlap for boundary context).
        self.kg_structural_batch_overlap: int | None = _opt_int("KG_STRUCTURAL_BATCH_OVERLAP")
        # Hard character cap per structural summary fed to the graph transformer.
        self.kg_structural_max_doc_chars: int | None = _opt_int("KG_STRUCTURAL_MAX_DOC_CHARS")

        # ── KG pipeline — technical track hyperparameters ────────────────
        # Per-file batching by default (1) for code-accurate entity extraction.
        self.kg_technical_batch_size: int | None = _opt_int("KG_TECHNICAL_BATCH_SIZE")
        self.kg_technical_batch_overlap: int | None = _opt_int("KG_TECHNICAL_BATCH_OVERLAP")
        # Higher cap than structural to preserve implementation detail.
        self.kg_technical_max_doc_chars: int | None = _opt_int("KG_TECHNICAL_MAX_DOC_CHARS")

        # ── Vertex AI RAG Engine (per-user empty corpus on registration) ─
        # GCP project/region for Vertex RAG. Defaults to the BigQuery project
        # and us-central1, matching the rest of the stack.
        self.vertex_project: str = (
            os.environ.get("VERTEX_PROJECT", "").strip() or self.bq_project_id
        )
        self.vertex_location: str = (
            os.environ.get("VERTEX_LOCATION", "").strip()
            or os.environ.get("GOOGLE_CLOUD_REGION", "").strip()
            or "us-central1"
        )
        # Optional Vertex publisher model used as the embedding model for the
        # RAG corpus (e.g. "publishers/google/models/text-embedding-005").
        # Leave blank to let the SDK pick its default.
        self.rag_embedding_publisher_model: str = os.environ.get(
            "RAG_EMBEDDING_PUBLISHER_MODEL", ""
        ).strip()
        # If true, /auth/register fails when the corpus cannot be created.
        # Defaults to false so transient Vertex outages don't lock out signups.
        self.rag_corpus_required_on_register: bool = (
            os.environ.get("RAG_CORPUS_REQUIRED_ON_REGISTER", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )


settings = Settings()
