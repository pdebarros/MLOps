# graphRAG agent

ADK agent that performs **two-source hybrid GraphRAG** combining:

1. **Neo4j experience knowledge graph** built by `experience_pipeline.py`
2. **Vertex AI RAG Engine corpus** of the user's GCS chunks

Both sources are tenant-scoped to the same normalised user id, and the agent
explicitly weights each source's contribution.

## Strategy

```
Query
  │
  ▼
[Step 0] assess_query_difficulty   ──────▶ chooses top_k & hop_depth
  │
  ├─── Graph source ─────────────────────────┐
  │                                          │
  │  [1a] vector_seed_search                 │
  │  [1b] fetch_neighborhood × top_k         │
  │  [1c] score_neighborhood × top_k         │
  │                                          │
  ├─── RAG source ───────────────────────────┤
  │                                          │
  │  [1d] rag_corpus_query                   │
  │  [1e] score_rag_evidence                 │
  │                                          │
  ▼                                          ▼
[Step 2] set_source_weights ◀────── (agent decides graph vs rag)
  │
  ▼
[Step 3] synthesize_evidence ────▶ weighted two-source evidence block
  │
  ▼
[Step 4] LLM answer grounded in synthesis
```

## Prerequisites

1. `experience_pipeline.py` has run for the target user — graph nodes have
   `embedding` populated and the `exp_entity_embedding` index exists.
2. **(Optional but recommended)** A Vertex AI RAG Engine corpus exists for
   the tenant. See "RAG configuration" below.
3. Application Default Credentials (`gcloud auth application-default login`)
   for local dev. In Agent Engine, the same code uses the attached service
   account automatically — no changes needed.

## Layout

| File | Purpose |
|------|---------|
| `agent.py` | Defines `root_agent` with all 9 tools wired up. |
| `tools.py` | Graph + RAG retrieval, scoring, weighting, synthesis. |
| `prompt.py` | System instruction enforcing the two-source workflow. |
| `config.py` | Reads Neo4j, Vertex AI, embedding, and RAG settings from `.env`. |
| `__init__.py` | Re-exports `root_agent`. |

## RAG configuration

Pick **one** primary mode in your `.env` (BigQuery lookup takes precedence when
configured).

### Mode 0 — BigQuery `rag_corpus_id` (recommended with gitai-upload-api)

When users register, the upload API creates an empty Vertex RAG corpus and
stores the corpus **numeric id** in BigQuery (`users.rag_corpus_id`). The
GraphRAG agent can resolve the full corpus resource name at query time using
the same **tenant id** passed into tools (the raw `user_id`).

**Option A — explicit table**

```
GRAPHRAG_BQ_USERS_TABLE_REF=my-gcp-project.gitai.users
GOOGLE_CLOUD_PROJECT=my-gcp-project
VERTEX_LOCATION=us-central1
```

**Option B — same env names as the upload API**

```
GRAPHRAG_RESOLVE_RAG_FROM_BQ=1
BQ_PROJECT_ID=my-gcp-project
BQ_DATASET=gitai
BQ_USERS_TABLE=users
GOOGLE_CLOUD_PROJECT=my-gcp-project
VERTEX_LOCATION=us-central1
```

The agent runs:

`SELECT rag_corpus_id FROM ... WHERE user_id = @tenant_id`

then builds:

`projects/<GOOGLE_CLOUD_PROJECT>/locations/<VERTEX_LOCATION>/ragCorpora/<rag_corpus_id>`

If `rag_corpus_id` is already a full `projects/.../ragCorpora/...` resource name,
it is used as-is.

The Agent Engine (or local) service account needs **BigQuery job user** + read
access on the users table (e.g. `roles/bigquery.dataViewer` on the dataset).

### Mode 1 — Per-tenant corpora (path template)

Each user has their own RAG corpus. Strict isolation, no metadata filtering.

```
VERTEX_RAG_CORPUS_TEMPLATE=projects/<PROJ>/locations/<LOC>/ragCorpora/gitai-{tenant}
```

The agent substitutes `{tenant}` with the normalised user id. So for user
`u_123` it queries `...ragCorpora/gitai-u_123`.

### Mode 2 — Shared corpus with metadata filter

A single corpus serves all tenants. Documents must have been uploaded with
metadata containing the tenant id.

```
VERTEX_RAG_CORPUS=projects/<PROJ>/locations/<LOC>/ragCorpora/<id>
VERTEX_RAG_TENANT_METADATA_KEY=tenant_id  # default
```

The agent applies a `Filter(metadata_filter='tenant_id="u_123"')` at query
time so cross-tenant chunks are excluded.

## Running

```bash
cd Ops_Project/
adk run KG_agent.graphRAG
```

When the agent prompts for the tenant id, supply the **raw user id** (e.g.
`d6f445be-...`); it is normalised internally to match
`experience_pipeline.normalize_user_id`.

### Example session

```
> How experienced is the user with async patterns?

[agent] assess_query_difficulty                → difficulty=hard, top_k=25, hop_depth=3
[agent] vector_seed_search                     → AsyncAwait, AsyncContextManager, ...
[agent] fetch_neighborhood × top seeds
[agent] score_neighborhood × top seeds         → 0.92, 0.85, 0.78, 0.55, 0.30
[agent] rag_corpus_query                       → 8 GCS chunks (mode=per_tenant)
[agent] score_rag_evidence                     → 0.65 (good context, no exact pattern match)
[agent] set_source_weights(0.7, 0.3,
        "Question is about skill demonstration which the graph captures
         explicitly via DEMONSTRATES edges.")  → graph 0.70, rag 0.30
[agent] synthesize_evidence(min_conf=0.4)
[agent] Final answer grounded in 4 graph neighborhoods + 6 RAG chunks
```

## Session state keys

The agent writes the following to `tool_context.state` so any tool can read
or override them:

| Key | Set by | Meaning |
|-----|--------|---------|
| `query`, `query_difficulty` | `assess_query_difficulty` | Original query + classification |
| `top_k`, `hop_depth` | `assess_query_difficulty` | Retrieval hyperparameters |
| `tenant_id` | `vector_seed_search` | Normalised user id |
| `seed_entity_ids`, `seed_metadata` | `vector_seed_search` | Top-k graph seeds |
| `neighborhoods` | `fetch_neighborhood` | Cached subgraphs keyed by seed id |
| `seed_scores` | `score_neighborhood` | Per-neighborhood confidence + rationale |
| `rag_chunks`, `rag_corpus`, `rag_mode` | `rag_corpus_query` | Retrieved chunks + source info |
| `rag_score` | `score_rag_evidence` | Overall RAG confidence + rationale |
| `source_weights` | `set_source_weights` | Normalised graph/rag weights + rationale |

## Tunables (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `GRAPHRAG_AGENT_MODEL` | `gemini-2.5-flash` | Model used by the agent. |
| `GRAPHRAG_MIN_TOP_K` / `GRAPHRAG_MAX_TOP_K` | 3 / 25 | Graph seed bounds. |
| `GRAPHRAG_MIN_HOPS` / `GRAPHRAG_MAX_HOPS` | 1 / 3 | Hop depth bounds. |
| `GRAPHRAG_MAX_NEIGHBOR_NODES` | 60 | Per-neighborhood node cap. |
| `GRAPHRAG_MAX_NEIGHBOR_EDGES` | 120 | Per-neighborhood edge cap. |
| `GRAPHRAG_RAG_TOP_K` | 8 | RAG chunks per query. |
| `GRAPHRAG_RAG_VECTOR_DISTANCE_THRESHOLD` | -1 (off) | RAG distance threshold. |
| `GRAPHRAG_NEO4J_DATABASE` | _empty_ | Logical DB name (AuraDB: leave empty). |
| `GRAPHRAG_BQ_USERS_TABLE_REF` | _unset_ | `project.dataset.users` for `rag_corpus_id` lookup. |
| `GRAPHRAG_RESOLVE_RAG_FROM_BQ` | `false` | With `BQ_*` / `GOOGLE_CLOUD_PROJECT`, compose table FQN. |
| `BQ_PROJECT_ID` / `BQ_DATASET` / `BQ_USERS_TABLE` | / `gitai` / `users` | Used when `GRAPHRAG_RESOLVE_RAG_FROM_BQ=1`. |
| `VERTEX_RAG_CORPUS_TEMPLATE` | _unset_ | Per-tenant corpus pattern with `{tenant}`. |
| `VERTEX_RAG_CORPUS` | _unset_ | Shared corpus (used if template unset). |
| `VERTEX_RAG_TENANT_METADATA_KEY` | `tenant_id` | Metadata key for shared-corpus filter. |

## Multi-tenancy guarantees

- **Graph**: every Cypher call carries the `:User_<uid>` label and a
  `tenantId = $uid` filter.
- **RAG (BigQuery mode)**: corpus id is loaded for that `user_id` only; each
  user maps to a distinct Vertex corpus resource.
- **RAG (per-tenant template mode)**: the corpus path itself contains the tenant id,
  so cross-tenant retrieval is structurally impossible.
- **RAG (shared mode)**: `Filter(metadata_filter='tenant_id="u_123"')` is
  applied at query time.

## Deployment notes

The agent uses Application Default Credentials for now. When deployed to
Vertex AI Agent Engine, attach a service account with:
- `roles/aiplatform.user` (Vertex AI RAG Engine retrieval)
- `roles/bigquery.jobUser` plus read on the users dataset when using Mode 0
- Network access to the Neo4j AuraDB instance
- Read access to the relevant GCS buckets if any tools later need them

No code changes are required for the credential switch — `vertexai.init()`
and `langchain_google_vertexai.VertexAIEmbeddings` both use ADC by default,
which on Agent Engine resolves to the attached SA.

For deployment, prefer:

```bash
python deploy.py --project <PROJECT> --region <REGION> --no-env-file
```

If you pass `--env-file`, avoid local path variables like `SSL_CERT_FILE`,
`REQUESTS_CA_BUNDLE`, and `CURL_CA_BUNDLE` (for example values under
`/Users/...`), because those paths do not exist in Agent Engine runtime.
