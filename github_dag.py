# dags/github_code_ingest_to_gcs.py

from __future__ import annotations

import io
import os
import re
import json
import tarfile
import hashlib
import zipfile
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import requests
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from google.cloud import storage

# ----------------------------
# Config
# ----------------------------

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "codebases-03-26")

# Where scraped training data will land
# Example:
# gs://codebases-03-26/github_corpus/raw/<owner>__<repo>__<sha>/...
RAW_PREFIX = os.getenv("GITHUB_RAW_PREFIX", "github_corpus/raw")
MANIFEST_PREFIX = os.getenv("GITHUB_MANIFEST_PREFIX", "github_corpus/manifests")

# Search seed(s)
DEFAULT_QUERIES = [
    "language:Python stars:>50 size:<50000 archived:false",
]

# File filters
ALLOWED_EXTENSIONS = {".py"}
EXCLUDED_DIRS = {
    ".git", ".github", "venv", ".venv", "env", "__pycache__", "node_modules",
    "dist", "build", "site-packages", ".mypy_cache", ".pytest_cache"
}
EXCLUDED_FILE_PATTERNS = [
    r"(^|/)(test_|tests?/)",
    r"(^|/)conftest\.py$",
    r"(^|/)setup\.py$",
    r"(^|/)manage\.py$",
]

# Keep only repos under permissive licenses if desired
# Set to False if you want to collect everything and review later
FILTER_PERMISSIVE_LICENSES = True
ALLOWED_LICENSE_KEYS = {
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc"
}

REQUEST_TIMEOUT = 30
MAX_REPOS_PER_RUN = 20
MAX_FILES_PER_REPO = 300
MAX_FILE_BYTES = 200_000  # skip giant files
MAX_ARCHIVE_BYTES = 100_000_000  # 100 MB
USER_AGENT = "ex-ide-ingestion-pipeline/1.0"

# Optional per-user/project routing
TARGET_USER_ID = os.getenv("TARGET_USER_ID", "github_training")

# ----------------------------
# Helpers
# ----------------------------

def github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers

def gh_get(url: str, params: Optional[dict] = None) -> requests.Response:
    r = requests.get(url, headers=github_headers(), params=params, timeout=REQUEST_TIMEOUT)
    if r.status_code >= 400:
        raise AirflowFailException(f"GitHub API error {r.status_code}: {url} -> {r.text[:500]}")
    return r

def is_excluded_path(path: str) -> bool:
    parts = path.split("/")
    if any(part in EXCLUDED_DIRS for part in parts):
        return True
    for pat in EXCLUDED_FILE_PATTERNS:
        if re.search(pat, path):
            return True
    return False

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def normalize_code(text: str) -> str:
    # Minimal normalization for dedupe
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip() + "\n"

def extract_archive_bytes(content: bytes, repo_full_name: str) -> List[Dict[str, Any]]:
    """
    Handles zip or tar.gz.
    Returns list of dicts with path/content.
    """
    files = []

    # Try zip first
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            for name in zf.namelist():
                if zf.getinfo(name).is_dir():
                    continue
                rel_path = "/".join(name.split("/")[1:])  # strip top-level repo folder
                if not rel_path:
                    continue
                ext = os.path.splitext(rel_path)[1].lower()
                if ext not in ALLOWED_EXTENSIONS or is_excluded_path(rel_path):
                    continue

                info = zf.getinfo(name)
                if info.file_size > MAX_FILE_BYTES:
                    continue

                raw = zf.read(name)
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                files.append({"path": rel_path, "content": normalize_code(text)})
        return files[:MAX_FILES_PER_REPO]
    except zipfile.BadZipFile:
        pass

    # Try tar.gz
    try:
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                rel_path = "/".join(member.name.split("/")[1:])
                if not rel_path:
                    continue
                ext = os.path.splitext(rel_path)[1].lower()
                if ext not in ALLOWED_EXTENSIONS or is_excluded_path(rel_path):
                    continue
                if member.size > MAX_FILE_BYTES:
                    continue

                f = tf.extractfile(member)
                if f is None:
                    continue
                raw = f.read()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue

                files.append({"path": rel_path, "content": normalize_code(text)})
        return files[:MAX_FILES_PER_REPO]
    except tarfile.ReadError:
        raise AirflowFailException(f"Could not read archive for {repo_full_name}")

def upload_text(bucket_name: str, blob_name: str, text: str, content_type: str = "text/plain") -> None:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string(text, content_type=content_type)

def upload_json(bucket_name: str, blob_name: str, payload: dict) -> None:
    upload_text(bucket_name, blob_name, json.dumps(payload, indent=2), "application/json")

# ----------------------------
# DAG
# ----------------------------

@dag(
    dag_id="github_code_ingest_to_gcs",
    start_date=datetime(2026, 4, 1),
    schedule="0 2 * * *",
    catchup=False,
    default_args={
        "owner": "ex-ide",
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["github", "gcs", "training", "kg"],
)
def github_code_ingest_to_gcs():

    @task
    def discover_repos() -> List[Dict[str, Any]]:
        """
        Uses GitHub search to find candidate repos.
        """
        all_items: List[Dict[str, Any]] = []

        queries = Variable.get("github_search_queries", default_var=json.dumps(DEFAULT_QUERIES))
        queries = json.loads(queries)

        for query in queries:
            resp = gh_get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": 25, "page": 1},
            ).json()

            items = resp.get("items", [])
            for item in items:
                all_items.append({
                    "full_name": item["full_name"],
                    "owner": item["owner"]["login"],
                    "repo": item["name"],
                    "default_branch": item["default_branch"],
                    "private": item["private"],
                    "fork": item["fork"],
                    "archived": item["archived"],
                    "html_url": item["html_url"],
                })

        # Deduplicate by full_name
        deduped = {}
        for item in all_items:
            deduped[item["full_name"]] = item

        results = list(deduped.values())[:MAX_REPOS_PER_RUN]
        logging.info("Discovered %d repositories", len(results))
        return results

    @task
    def filter_repos(repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes forks, archived repos, and optionally non-permissive licenses.
        """
        filtered = []

        for repo in repos:
            if repo["private"] or repo["fork"] or repo["archived"]:
                continue

            meta = gh_get(f"https://api.github.com/repos/{repo['full_name']}").json()
            license_info = meta.get("license")
            license_key = license_info["key"] if license_info else None

            if FILTER_PERMISSIVE_LICENSES and license_key not in ALLOWED_LICENSE_KEYS:
                continue

            filtered.append({
                **repo,
                "license_key": license_key,
                "stargazers_count": meta.get("stargazers_count", 0),
                "pushed_at": meta.get("pushed_at"),
                "size": meta.get("size"),
            })

        logging.info("Filtered to %d repositories", len(filtered))
        return filtered

    @task
    def download_and_extract(repos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Downloads repo archives and extracts eligible Python files.
        """
        processed = []

        for repo in repos:
            full_name = repo["full_name"]

            # Resolve latest commit SHA on default branch
            branch_meta = gh_get(
                f"https://api.github.com/repos/{full_name}/branches/{repo['default_branch']}"
            ).json()
            sha = branch_meta["commit"]["sha"]

            archive_url = f"https://api.github.com/repos/{full_name}/zipball/{sha}"
            r = requests.get(archive_url, headers=github_headers(), timeout=REQUEST_TIMEOUT)
            if r.status_code >= 400:
                raise AirflowFailException(f"Failed archive download for {full_name}: {r.status_code}")

            archive_size = len(r.content)
            if archive_size > MAX_ARCHIVE_BYTES:
                logging.warning("Skipping %s: archive too large (%d bytes)", full_name, archive_size)
                continue

            files = extract_archive_bytes(r.content, full_name)

            # Dedupe file contents within repo
            seen_hashes = set()
            cleaned_files = []
            for f in files:
                content_hash = sha256_text(f["content"])
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
                cleaned_files.append({
                    "path": f["path"],
                    "content": f["content"],
                    "content_hash": content_hash,
                    "bytes": len(f["content"].encode("utf-8")),
                })

            if not cleaned_files:
                continue

            processed.append({
                **repo,
                "sha": sha,
                "files": cleaned_files,
                "file_count": len(cleaned_files),
            })

        logging.info("Prepared %d repositories for upload", len(processed))
        return processed

    @task
    def upload_to_gcs(repos_with_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Upload each file and a repo manifest to GCS.
        """
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)

        uploaded_manifests = []

        for repo in repos_with_files:
            repo_key = f"{repo['owner']}__{repo['repo']}__{repo['sha']}"
            base_prefix = f"{RAW_PREFIX}/{repo_key}"

            manifest = {
                "repo_full_name": repo["full_name"],
                "owner": repo["owner"],
                "repo": repo["repo"],
                "sha": repo["sha"],
                "default_branch": repo["default_branch"],
                "html_url": repo["html_url"],
                "license_key": repo.get("license_key"),
                "stargazers_count": repo.get("stargazers_count"),
                "pushed_at": repo.get("pushed_at"),
                "file_count": repo["file_count"],
                "target_user_id": TARGET_USER_ID,
                "files": [],
                "ingested_at": datetime.utcnow().isoformat() + "Z",
            }

            for f in repo["files"]:
                blob_path = f"{base_prefix}/{f['path']}"
                blob = bucket.blob(blob_path)
                blob.upload_from_string(f["content"], content_type="text/x-python")

                manifest["files"].append({
                    "gcs_path": f"gs://{GCS_BUCKET}/{blob_path}",
                    "path": f["path"],
                    "content_hash": f["content_hash"],
                    "bytes": f["bytes"],
                })

            manifest_blob = f"{MANIFEST_PREFIX}/{repo_key}.json"
            upload_json(GCS_BUCKET, manifest_blob, manifest)
            uploaded_manifests.append({
                "repo_full_name": repo["full_name"],
                "manifest_gcs_path": f"gs://{GCS_BUCKET}/{manifest_blob}",
                "repo_prefix": f"gs://{GCS_BUCKET}/{base_prefix}",
                "file_count": repo["file_count"],
            })

        logging.info("Uploaded %d manifests", len(uploaded_manifests))
        return uploaded_manifests

    @task
    def write_run_summary(uploaded: List[Dict[str, Any]]) -> str:
        summary = {
            "run_at": datetime.utcnow().isoformat() + "Z",
            "repos_uploaded": len(uploaded),
            "items": uploaded,
        }
        run_key = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        blob_name = f"{MANIFEST_PREFIX}/runs/{run_key}.json"
        upload_json(GCS_BUCKET, blob_name, summary)
        return f"gs://{GCS_BUCKET}/{blob_name}"

    discover_repos() >> filter_repos() >> download_and_extract() >> upload_to_gcs() >> write_run_summary()

github_code_ingest_to_gcs()