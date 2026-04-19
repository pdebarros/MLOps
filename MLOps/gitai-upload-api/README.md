# git-ai Upload API (BigQuery Auth + GCS)

FastAPI service for:
- user registration/login backed by BigQuery table
- authenticated Python-change ingest from `tracking_ui.py`
- storing each ingest payload in GCS under user-scoped paths

## 1) Create BigQuery dataset/table (CLI)

```bash
export PROJECT_ID="your-gcp-project-id"
export DATASET="gitai"
export TABLE="users"

bq --project_id="$PROJECT_ID" mk --dataset "$PROJECT_ID:$DATASET"

bq --project_id="$PROJECT_ID" mk --table "$PROJECT_ID:$DATASET.$TABLE" \
user_id:STRING,email:STRING,password_hash:STRING,created_at:TIMESTAMP,last_login_at:TIMESTAMP,is_active:BOOL,session_token_hash:STRING,session_expires_at:TIMESTAMP,total_upload_requests:INT64,total_uploaded_lines:INT64
```

Optional dedupe/cleanup (manual):
```bash
bq --project_id="$PROJECT_ID" query --use_legacy_sql=false \
'SELECT email, COUNT(*) c FROM `'$PROJECT_ID'.'$DATASET'.'$TABLE'` GROUP BY email HAVING c > 1'
```

## 2) Local run

```bash
cd gitai-upload-api
cp .env.example .env
# Edit .env with API_KEY, GCS_BUCKET, BQ_PROJECT_ID, etc.
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## 3) Cloud Run deploy

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export SERVICE="gitai-upload-api"

gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE
gcloud run deploy $SERVICE \
  --image gcr.io/$PROJECT_ID/$SERVICE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars API_KEY=your-global-api-key,GCS_BUCKET=your-bucket,GCS_OBJECT_PREFIX=gitai-python,BQ_PROJECT_ID=$PROJECT_ID,BQ_DATASET=gitai,BQ_USERS_TABLE=users,SESSION_TTL_HOURS=24 \
  --service-account YOUR_RUN_SA@$PROJECT_ID.iam.gserviceaccount.com
```

Grant service account access:
- BigQuery: `roles/bigquery.dataEditor` on dataset + `roles/bigquery.jobUser`
- GCS bucket: `roles/storage.objectAdmin` (or narrower create/get)

## 4) API

- `POST /auth/register` -> creates user row (generated `user_id`), returns session token
- `POST /auth/login` -> verifies password, returns session token
- `POST /ingest/python` -> requires headers:
  - `X-User-Email`
  - `X-Session-Token`
  - `X-API-Key` (only if API_KEY is configured server-side)

Payload shape is `PythonIngest` in `app/main.py`.

## 5) Troubleshooting

### “streaming buffer” UPDATE errors
User rows must be inserted with **SQL `INSERT` (DML)**, not the streaming `insert_rows_json` API, or immediate `UPDATE` for sessions can fail. This service uses DML `INSERT` for new registrations.

If you **already** inserted test users via streaming, those rows can block `UPDATE` until BigQuery commits them (often up to ~90 minutes). Fix: use a **new** `users` table name in `.env` (`BQ_USERS_TABLE=users_v2`) and recreate the schema, or wait and retry.

### ADC “no quota project” warning (local dev)
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

Setting `BQ_PROJECT_ID` in `.env` also sets `GOOGLE_CLOUD_PROJECT` for client libraries when the app loads.
