# git-ai Upload API

FastAPI service that powers the gitai desktop app:

- User registration / login backed by BigQuery
- Authenticated Python file ingest + GCS storage
- Triggers the KG pipeline (multi-tenant AuraDB) after every file submission

## Architecture

```
gitai app ──► Cloud Run (upload API) ──► GCS  (raw Python files)
                                    └──► Cloud Run Job (kg-pipeline) ──► AuraDB
```

The **upload API** is a small, stateless FastAPI service (no KG deps). After each file upload it creates a **Cloud Run Job execution** that runs `prod_pipeline.py` with the user's ID. Job status can be polled via `/kg/jobs/{job_id}`.

---

## 1. BigQuery — create the users table

```bash
export PROJECT_ID="your-gcp-project-id"
export DATASET="gitai"

bq --project_id="$PROJECT_ID" mk --dataset "$PROJECT_ID:$DATASET"

bq --project_id="$PROJECT_ID" mk --table "$PROJECT_ID:$DATASET.users" \
  user_id:STRING,email:STRING,password_hash:STRING,created_at:TIMESTAMP,\
last_login_at:TIMESTAMP,is_active:BOOL,session_token_hash:STRING,\
session_expires_at:TIMESTAMP,total_upload_requests:INT64,total_uploaded_lines:INT64,\
rag_corpus_id:STRING
```

`rag_corpus_id` stores the per-user Vertex AI RAG Engine corpus **numeric id**
(the last segment of the resource name) that's created at registration.
Reconstruct the full resource name when calling Vertex as
`projects/<VERTEX_PROJECT>/locations/<VERTEX_LOCATION>/ragCorpora/<id>`.

**RAG backend:** registration creates corpora with
`EmbeddingModelConfig(publisher_model=text-embedding-004)` in
`VERTEX_LOCATION` (default `europe-west4`, Netherlands), which provisions
**Spanner-backed** RAG rather than Serverless Vector Search. Override
`VERTEX_LOCATION` or `RAG_EMBEDDING_PUBLISHER_MODEL` via env if needed.

To migrate an existing table that doesn't have the column yet, add it:

```bash
bq --project_id="$PROJECT_ID" update \
  --schema=user_id:STRING,email:STRING,password_hash:STRING,created_at:TIMESTAMP,last_login_at:TIMESTAMP,is_active:BOOL,session_token_hash:STRING,session_expires_at:TIMESTAMP,total_upload_requests:INT64,total_uploaded_lines:INT64,rag_corpus_id:STRING \
  "$PROJECT_ID:$DATASET.users"
```

If you previously added a `rag_corpus_name STRING` column from an earlier
revision of this guide, rename it in place (BigQuery supports column
renames):

```bash
bq query --use_legacy_sql=false --project_id="$PROJECT_ID" \
  "ALTER TABLE \`$PROJECT_ID.$DATASET.users\` RENAME COLUMN rag_corpus_name TO rag_corpus_id"
```

---

## 2. Local development

```bash
cd gitai-upload-api
cp .env.example .env
# Fill in .env (see section 5 for required values)
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

For local dev the KG pipeline runs as a background subprocess.  
Set `KG_PIPELINE_PYTHON` and `KG_PIPELINE_SCRIPT` in `.env`.

---

## 3. Cloud Run deployment

### 3a. IAM service accounts

Create (or reuse) two service accounts:

| Account | Purpose |
|---|---|
| `gitai-run-sa` | Runs the upload API Cloud Run service |
| `gitai-kg-sa` | Runs the KG pipeline Cloud Run Job |

```bash
export PROJECT_ID="your-project"
export REGION="us-central1"

# Upload API service account
gcloud iam service-accounts create gitai-run-sa \
  --display-name="gitai Upload API"

# KG pipeline service account
gcloud iam service-accounts create gitai-kg-sa \
  --display-name="gitai KG Pipeline Job"

# Roles for upload API SA
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gitai-run-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gitai-run-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.jobUser"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gitai-run-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gitai-run-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.developer"   # needed to trigger Cloud Run Job executions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gitai-run-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"  # needed to create per-user Vertex RAG corpora

# Roles for KG pipeline SA
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gitai-kg-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gitai-kg-sa@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### 3b. Build + push the upload API image

```bash
cd gitai-upload-api

gcloud builds submit \
  --tag gcr.io/$PROJECT_ID/gitai-upload-api \
  --project $PROJECT_ID
```

### 3c. Deploy the upload API to Cloud Run

```bash
export CORS_ORIGINS="tauri://localhost,https://app.example.com"
export API_KEY="your-secret-key"           # shared with the gitai app
export GCS_BUCKET="your-gitai-diff-bucket" # for Python diffs (ingest/python)
export PYTHON_FILES_BUCKET="codebases-04-03-26"  # for full files (ingest/python-file)
export KG_JOB_NAME="kg-pipeline-job"       # Cloud Run Job name (created in step 3e)

gcloud run deploy gitai-upload-api \
  --image gcr.io/$PROJECT_ID/gitai-upload-api \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 60s \
  --service-account gitai-run-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars \
API_KEY=$API_KEY,\
GCS_BUCKET=$GCS_BUCKET,\
PYTHON_FILES_BUCKET=$PYTHON_FILES_BUCKET,\
BQ_PROJECT_ID=$PROJECT_ID,\
BQ_DATASET=gitai,\
BQ_USERS_TABLE=users,\
SESSION_TTL_HOURS=24,\
CORS_ALLOW_ORIGINS=$CORS_ORIGINS,\
CLOUD_RUN_REGION=$REGION,\
CLOUD_RUN_KG_JOB_NAME=$KG_JOB_NAME
```

### 3d. Build + push the KG pipeline Job image

```bash
cd ../Ops_Project/KG_agent

gcloud builds submit \
  --tag gcr.io/$PROJECT_ID/kg-pipeline-job \
  --project $PROJECT_ID
```

### 3e. Create the KG pipeline Cloud Run Job

Secrets (Neo4j password, API keys) are best stored in Secret Manager and mounted as env vars.
For a quick start you can pass them directly with `--set-env-vars`:

```bash
export NEO4J_URI="neo4j+s://XXXX.databases.neo4j.io"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-neo4j-password"
export GCS_BUCKET_NAME="codebases-04-03-26"
export GOOGLE_CLOUD_PROJECT=$PROJECT_ID
export GOOGLE_CLOUD_REGION=$REGION
export GEMINI_MODEL="gemini-2.0-flash"   # or whichever model you use
export VERTEX_GEMINI_MODEL="gemini-2.0-flash"

gcloud run jobs create kg-pipeline-job \
  --image gcr.io/$PROJECT_ID/kg-pipeline-job \
  --region $REGION \
  --service-account gitai-kg-sa@$PROJECT_ID.iam.gserviceaccount.com \
  --task-timeout 3600s \
  --max-retries 1 \
  --set-env-vars \
NEO4J_URI=$NEO4J_URI,\
NEO4J_USER=$NEO4J_USER,\
NEO4J_PASSWORD=$NEO4J_PASSWORD,\
GCS_BUCKET_NAME=$GCS_BUCKET_NAME,\
GOOGLE_CLOUD_PROJECT=$GOOGLE_CLOUD_PROJECT,\
GOOGLE_CLOUD_REGION=$GOOGLE_CLOUD_REGION,\
GEMINI_MODEL=$GEMINI_MODEL,\
VERTEX_GEMINI_MODEL=$VERTEX_GEMINI_MODEL
```

To update an existing job's env vars:
```bash
gcloud run jobs update kg-pipeline-job \
  --region $REGION \
  --update-env-vars NEO4J_PASSWORD=new-password
```

### 3f. Update the gitai app to point to Cloud Run

In the gitai desktop app settings, replace `http://localhost:8080` with the Cloud Run service URL:

```
https://gitai-upload-api-XXXX-uc.a.run.app
```

---

## 4. Automated deploys with Cloud Build

Commit `.cloudbuild.yaml` to your repo and connect a Cloud Build trigger. The trigger substitution values map to the env vars in step 3c.

---

## 5. Environment variable reference

| Variable | Required | Description |
|---|---|---|
| `API_KEY` | No | Global API gate key (shared with app) |
| `GCS_BUCKET` | Yes | Bucket for Python diff payloads |
| `PYTHON_FILES_BUCKET` | Yes | Bucket for full Python source files (must match KG pipeline) |
| `BQ_PROJECT_ID` | Yes | GCP project for BigQuery |
| `BQ_DATASET` | No | BigQuery dataset name (default: `gitai`) |
| `BQ_USERS_TABLE` | No | Users table name (default: `users`) |
| `SESSION_TTL_HOURS` | No | Login token lifetime (default: `24`) |
| `CORS_ALLOW_ORIGINS` | No | Comma-separated allowed origins |
| `CLOUD_RUN_KG_JOB_NAME` | Yes (prod) | Name of the KG pipeline Cloud Run Job |
| `CLOUD_RUN_REGION` | No | Region of the Job (default: `us-central1`) |
| `CLOUD_RUN_PROJECT` | No | GCP project for the Job (default: `BQ_PROJECT_ID`) |
| `KG_PIPELINE_SCRIPT` | Local only | Path to `prod_pipeline.py` |
| `KG_PIPELINE_PYTHON` | Local only | Python interpreter with KG deps |
| `KG_PIPELINE_DATABASE` | No | Neo4j database name (leave blank for AuraDB default) |
| `KG_STRUCTURAL_BATCH_SIZE` | No | Structural KG batch size override |
| `KG_TECHNICAL_BATCH_SIZE` | No | Technical KG batch size override |
| `VERTEX_PROJECT` | No | GCP project for Vertex RAG Engine (defaults to `BQ_PROJECT_ID`) |
| `VERTEX_LOCATION` | No | Vertex region for the per-user RAG corpus (default: `europe-west4` — Spanner RAG) |
| `RAG_EMBEDDING_PUBLISHER_MODEL` | No | Publisher embedding model (default: `publishers/google/models/text-embedding-004`) |
| `RAG_CORPUS_REQUIRED_ON_REGISTER` | No | If `1/true`, registration fails when the corpus cannot be created (default: lenient) |

---

## 6. API endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/auth/register` | App key | Create account |
| `POST` | `/auth/login` | App key | Get session token |
| `POST` | `/ingest/python` | Session | Upload Python diffs (line tracking) |
| `POST` | `/ingest/python-file` | Session | Upload full Python files, triggers KG pipeline |
| `GET` | `/kg/jobs/{job_id}` | Session | Poll KG pipeline job status |
| `GET` | `/health` | — | Health check |

---

## 7. Troubleshooting

**BigQuery streaming buffer UPDATE errors**  
User rows must use SQL DML `INSERT`, not the streaming API, or immediate `UPDATE` can fail. This service uses DML. If you already have streaming-inserted rows, wait ~90 minutes or use a new `BQ_USERS_TABLE` name.

**ADC "no quota project" warning**  
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT_ID
```

**Cloud Run Job not found**  
Ensure the Job was created in the same `CLOUD_RUN_REGION` and that the upload API's service account has `roles/run.developer` on the project.

**KG pipeline `skipped` (all summaries already ingested)**  
This is correct incremental behavior. Upload new `.py` files or clear the GCS summary cache to force re-ingestion:
```bash
gsutil rm -r gs://BUCKET/USER_ID/summaries_structural/
gsutil rm -r gs://BUCKET/USER_ID/summaries_technical/
```
