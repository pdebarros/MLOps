# KG pipeline: `pipeline2`, `tools2`, `kg_reconstruct_test`

This folder implements an end-to-end flow: **load Python from GCS → summarize with Gemini (Google ADK) → cache summaries in GCS → extract a knowledge graph → persist to Neo4j**. Configuration is centralized in **`config.py`** (environment variables and optional `.env` files).

---

## Prerequisites

- Python 3.10+ and project dependencies (`pip install -r ../requirements.txt` from the repo root).
- **Google Cloud**
  - Application Default Credentials (e.g. `gcloud auth application-default login`) for GCS, Vertex AI, and ADK where applicable.
  - Access to the GCS bucket you configure (`GCS_BUCKET_NAME`, default `codebases-03-26`).
- **Neo4j** reachable from this machine (`NEO4J_*` in `.env`).
- **Optional:** `python-dotenv` so `config.py` can load `KG_agent/.env` or the parent folder’s `.env`.

---

## Configuration (`config.py`)

Importing `config` loads `.env` from **`KG_agent/.env`** then **`../.env`** (project root).

| Area | Variables (examples) |
|------|----------------------|
| Neo4j | `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` |
| GCS / summarization | `GCS_BUCKET_NAME`, `GEMINI_MODEL`, `SUMMARY_MAX_PARALLEL` |
| KG graph LLM | `KG_GRAPH_BACKEND` (`vertex` or `huggingface`), `VERTEX_GEMINI_MODEL` / `GEMINI_GRAPH_MODEL`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_REGION` or `VERTEX_LOCATION`, `VERTEX_TEMPERATURE`, `VERTEX_MAX_TOKENS` |
| HF (if `KG_GRAPH_BACKEND=huggingface`) | `HF_GRAPH_MODEL`, `HF_MAX_NEW_TOKENS`, `HF_TEMPERATURE` |
| KG chunking / schema | `KG_MAX_DOC_CHARS`, `KG_BATCH_SIZE`, `KG_BATCH_OVERLAP`, `ALLOWED_NODE_TYPES`, `ALLOWED_REL_TYPES`, `KG_VERTEX_IGNORE_TOOL_USAGE` |

Use `from config import Config` (or `config`) to read resolved values.

---

## `pipeline2.py` — Summarize code + build KG

**What it does**

1. Lists every **`*.py`** object under `gs://<bucket>/<user_id>/`, **excluding** `.../<user_id>/summaries/...` (those are cached JSON, not source).
2. For each `.py` file, checks for a **completed** summary JSON in GCS. If **all** files already have valid summaries, it **skips** the ADK summarization step and loads summaries from GCS only.
3. Otherwise it summarizes only **missing** or **failed** files (parallelism from `SUMMARY_MAX_PARALLEL`), **uploads** one JSON per file, merges with existing summaries, then runs the KG step.
4. Calls **`build_kg_and_push_to_neo4j`** from `tools2` on the full list of summary records.

**GCS layout**

| Path | Role |
|------|------|
| `gs://<bucket>/<user_id>/path/to/file.py` | Source code |
| `gs://<bucket>/<user_id>/summaries/path/to/file.py.json` | Cached summary (`file`, `summary`, `status`, optional `error`) |

**Run**

```bash
cd KG_agent
python pipeline2.py --user-id u_123
```

`--user-id` is the folder under the bucket (e.g. `u_123` → `gs://codebases-03-26/u_123/`).

**Auth**

- GCS: ADC (same as other Google client libraries).
- Summaries: **Gemini model name** from `GEMINI_MODEL` via Google ADK `Agent`.

---

## `tools2.py` — Knowledge graph extraction + Neo4j

**What it does**

- Takes a **list of dicts** like the pipeline produces: `status == "completed"` and a `summary` string per file.
- Builds LangChain **`Document`**s (with optional batching via `KG_BATCH_SIZE` / `KG_BATCH_OVERLAP` for cross-file context).
- Runs **`LLMGraphTransformer`**:
  - Default **`KG_GRAPH_BACKEND=vertex`**: **Vertex AI Gemini** (`ChatVertexAI`) with ADC.
  - Alternative: **`huggingface`** local model via `transformers` + `HuggingFacePipeline` (no API calls; heavier locally).
- Writes nodes and relationships to Neo4j with **`MERGE`** (upsert by entity `id` and relationship type property), using labels **`:Entity`** and **`:REL`**.

**Main entry**

```python
from tools2 import build_kg_and_push_to_neo4j

result = build_kg_and_push_to_neo4j(results)  # list of summary dicts from pipeline2
```

**Persistence note**

Repeated runs **merge** into the same graph; they do **not** delete nodes or edges that are no longer mentioned. Plan for occasional cleanup or namespacing if you need strict per-project isolation.

---

## `kg_reconstruct_test.py` — Subgraph → text (evaluation / QA)

**What it does**

- Loads a **k-hop** neighborhood around Neo4j **`:Entity`** nodes whose **`id`** matches `--needle` (substring) or `--exact-id`.
- Serializes nodes and edges as bullet text.
- Optionally calls **Vertex Gemini** (same `Config` as `tools2`) with a fixed prompt: short summary of the slice, **only** facts supported by the graph, and explicit uncertainty.

**Run**

Run the **script path** directly (avoid `python -m KG_agent` if `KG_agent/__init__.py` imports optional agents that may break):

```bash
# From repo root
python KG_agent/kg_reconstruct_test.py --needle pipeline.py --k 2 --dry-run
python KG_agent/kg_reconstruct_test.py --exact-id "path/to/entity_id" --k 3
```

| Flag | Meaning |
|------|---------|
| `--needle` | Substring match on `Entity.id` |
| `--exact-id` | Exact `Entity.id` |
| `--k` | Hop depth (1–10; `0..k` includes the start node) |
| `--max-starts` | Cap how many start nodes match the needle |
| `--dry-run` | Print serialized graph only; no Vertex call |

---

## Related files

- **`config.py`** — Shared settings for these scripts.
- **`requirements.txt`** (repo root) — Dependencies including `google-adk`, `google-cloud-storage`, `neo4j`, `langchain-*`, `langchain-google-vertexai`, etc.
