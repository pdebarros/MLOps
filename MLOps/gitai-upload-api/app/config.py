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


class Settings:
    """Populated from environment (Cloud Run) or a local .env file (via python-dotenv)."""

    def __init__(self) -> None:
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


settings = Settings()
