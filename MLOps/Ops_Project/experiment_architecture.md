# Technical Architecture Document: `experiment.py` and Dependency Stack

## Title Page

**Document Title:** End-to-End Technical Architecture of `experiment.py`  
**Project:** `Ops_Project`  
**Primary Entrypoint:** `experiment.py`  
**Primary Dependencies:** `KG_agent/pipeline3.py`, `KG_agent/pipeline2.py`, `KG_agent/tools2.py`, `KG_agent/config.py`, `eval_agent` modules  
**Prepared For:** Manual engineering review, reproducibility, and tuning analysis  
**Date:** 2026-04-16

---

## 1. Executive Overview

`experiment.py` is an orchestration script that runs a full knowledge-graph evaluation lifecycle and records results in MLflow. It coordinates four major phases:

1. **Graph construction** using `pipeline3.py` into a **fresh Neo4j logical database** per run.
2. **Graph metadata extraction** (counts + schema-like distributions) directly from Neo4j.
3. **Evaluation execution** through an ADK-based evaluation agent.
4. **Score ingestion** from GCS and logging as MLflow metrics/artifacts.

Its architectural role is **integration orchestration**, not transformation logic itself. Actual summarization and KG extraction are delegated to the `KG_agent` modules.

---

## 2. Scope and Module Boundaries

### In-scope for this document
- `experiment.py`
- `KG_agent/config.py`
- `KG_agent/pipeline3.py`
- `KG_agent/pipeline2.py` (shared functions used by pipeline3)
- `KG_agent/tools2.py`

### Out-of-scope (referenced but externalized)
- Full internals of `eval_agent.agent` and its tool implementations
- MLflow backend internals
- GCP infra provisioning details

---

## 3. Runtime Entry and Invocation

### CLI contract (`experiment.py`)
- `--user-id` (**required**): GCS namespace identity / prefix basis
- `--experiment-name` (**optional**): MLflow experiment name; defaults to `kg_pipeline_eval_fix`

Execution:

```bash
cd Ops_Project
python experiment.py --user-id u_123 --experiment-name my_exp
```

Entrypoint function:

```python
def main() -> None:
    ...
    asyncio.run(run_experiment(args.user_id, args.experiment_name))
```

---

## 4. High-Level Dataflow

1. Normalize identity and create randomized Neo4j DB name.
2. Start MLflow run and log static run metadata.
3. Ensure Neo4j database exists (best effort).
4. Run dual-track KG pipeline (`pipeline3`):
   - Structural summaries + graph extraction
   - Technical summaries + graph extraction
5. Query resulting Neo4j DB for scoped metadata.
6. Run eval agent session using the newly created DB.
7. Read latest scoring row from GCS.
8. Log all artifacts/metrics into MLflow.

---

## 5. Detailed Walkthrough of `experiment.py`

### 5.1 Import and path bootstrap

`experiment.py` prepends `Ops_Project/KG_agent` and root to `sys.path` so local modules can be imported via short names (`config`, `pipeline2`, `pipeline3`).

### 5.2 Neo4j database naming strategy

- `neo4j_experiment_database_name()` generates a 6-char DB name where first char is alpha.
- `_sanitize_neo4j_database_name()` enforces Neo4j naming constraints:
  - allowed chars `[A-Za-z0-9_]`
  - must start with a letter
  - max length capped to 48

This gives per-run DB isolation.

### 5.3 Database creation preflight

`ensure_neo4j_database_exists(database)` attempts:

```cypher
CREATE DATABASE `<name>` IF NOT EXISTS
```

against system DB. Failures are logged as warnings and do not hard-stop the run.

### 5.4 MLflow model context logging

`log_kg_model_params()` logs the active summarization and graph extraction model config from `KG_agent.config.Config`:
- `summary_llm_model`
- `kg_graph_backend`
- `kg_vertex_graph_model`
- `kg_hf_graph_model`

### 5.5 KG pipeline execution

`run_experiment()` calls:

```python
pipeline_out = await kg_pipeline.run_pipeline3(user_id, neo4j_database=neo4j_db)
```

and logs:
- full payload as `kg_pipeline_result.json`
- extracted metrics from both tracks (summary counts, nodes/edges written, status)

### 5.6 Neo4j metadata logging

`neo4j_graph_metadata_for_user_prefix` computes, scoped by `Entity.id STARTS WITH "<normalized_user_id>/"`:
- total entity nodes
- total `:REL` edges
- histogram of `r.type`
- histogram of `n.kind`

Then `log_neo4j_schema_to_mlflow` emits:
- artifact `neo4j_graph_schema_counts.json`
- per-type metrics (`neo4j_rel_type_*`, `neo4j_entity_kind_*`)

### 5.7 Eval-agent invocation

`run_eval_agent_session` creates an ADK `Runner` with `eval_agent.root_agent`, injects a strict prompt requiring all graph query tools to use the generated `neo4j_database`, and captures final text output.

### 5.8 Score ingestion from GCS

`fetch_latest_scoring_record(user_id)` reads final NDJSON row from the eval scoring blob and logs:
- artifact: `eval_scoring_record.json`
- scalar metrics when present:
  - `eval_criterion_1_score`
  - `eval_criterion_2_score`
  - `eval_overall_score`
  - `eval_samples_evaluated`

---

## 6. Dependency Architecture

### 6.1 `KG_agent/config.py` (global config snapshot)

#### Responsibility
Centralized environment-backed config + `.env` loading.

#### Load order
1. `KG_agent/.env`
2. `Ops_Project/.env`

`Config` fields are class attributes evaluated at import time.

#### Major config domains
- Neo4j connectivity
- GCS + summarization settings
- KG backend/model selection
- KG chunking/batching/schema constraints
- Legacy model settings

---

### 6.2 `KG_agent/pipeline3.py` (dual-track orchestrator)

#### Responsibility
Run two independent summary-to-graph tracks and merge output into same Neo4j DB.

#### Track behavior
- **Structural track**
  - architecture-oriented summarization instructions
  - larger batch/overlap defaults
  - writes with `extra_entity_props={"kg_track_structural": True}`
- **Technical track**
  - implementation-oriented summarization instructions
  - tighter batching defaults
  - writes with `extra_entity_props={"kg_track_technical": True}`

#### Caching model
Each track uses a separate summary cache folder:
- `summaries_structural`
- `summaries_technical`

Missing/invalid summaries are regenerated; existing valid summaries are reused.

---

### 6.3 `KG_agent/pipeline2.py` (shared summarization infrastructure)

`pipeline3` reuses these primitives:
- user-id normalization
- GCS `.py` listing and reads
- summary existence/validity checks
- summary upload/load
- merge policy for cached + fresh results
- parallel summarization runtime

#### Concurrency control
`run_parallel_summaries` uses `asyncio.Semaphore(max_parallel)` where `max_parallel` comes from `SUMMARY_MAX_PARALLEL`.

---

### 6.4 `KG_agent/tools2.py` (graph extraction and Neo4j persistence)

#### Extraction path
1. select completed summaries
2. truncate per summary (`KG_MAX_DOC_CHARS` or override)
3. chunk summaries (`KG_BATCH_SIZE`, `KG_BATCH_OVERLAP` or overrides)
4. convert each chunk to graph documents via `LLMGraphTransformer`
5. persist to Neo4j using MERGE semantics

#### Neo4j schema conventions
- Node label: `:Entity` with key `id`
- Node semantic type stored in `kind`
- Edge type materialized as `:REL {type: <semantic_relation>}`

#### Important merge behavior
Node writes use:

```cypher
MERGE (n:Entity {id: $id})
SET n += $props
```

This means repeated writes by multiple tracks augment the same node, enabling both provenance flags to be true simultaneously.

---

## 7. Call Graph (Logical)

`experiment.py::main`  
→ `run_experiment`  
→ `ensure_neo4j_database_exists`  
→ `pipeline3.run_pipeline3`  
→ `pipeline3.run_single_track(structural)`  
→ `pipeline2.run_parallel_summaries` (if needed)  
→ `tools2.build_kg_and_push_to_neo4j`  
→ `pipeline3.run_single_track(technical)`  
→ `pipeline2.run_parallel_summaries` (if needed)  
→ `tools2.build_kg_and_push_to_neo4j`  
→ back to `experiment.py`  
→ `neo4j_graph_metadata_for_user_prefix`  
→ `run_eval_agent_session`  
→ `fetch_latest_scoring_record`  
→ MLflow run close

---

## 8. Tunable vs Non-Tunable Parameters

### 8.1 Tunable (CLI)

- `experiment.py`
  - `--user-id`
  - `--experiment-name`
- `pipeline3.py` (standalone mode)
  - `--user-id`
  - `--neo4j-database`

---

### 8.2 Tunable (Environment / `.env` / shell)

#### Summarization layer
- `GEMINI_MODEL`
- `SUMMARY_MAX_PARALLEL`

#### Graph extraction backend/model
- `KG_GRAPH_BACKEND`
- Vertex path:
  - `VERTEX_GEMINI_MODEL` (or fallback `GEMINI_GRAPH_MODEL`)
  - `VERTEX_LOCATION`
  - `VERTEX_TEMPERATURE`
  - `VERTEX_MAX_TOKENS`
  - `KG_VERTEX_IGNORE_TOOL_USAGE`
- HF path:
  - `HF_GRAPH_MODEL`
  - `HF_MAX_NEW_TOKENS`
  - `HF_TEMPERATURE`

#### KG shaping
- `KG_MAX_DOC_CHARS`
- `KG_BATCH_SIZE`
- `KG_BATCH_OVERLAP`
- `ALLOWED_NODE_TYPES`
- `ALLOWED_REL_TYPES`

#### Pipeline3 per-track KG tuning
- `PIPELINE3_STRUCTURAL_KG_BATCH`
- `PIPELINE3_STRUCTURAL_KG_OVERLAP`
- `PIPELINE3_STRUCTURAL_MAX_DOC_CHARS`
- `PIPELINE3_TECHNICAL_KG_BATCH`
- `PIPELINE3_TECHNICAL_KG_OVERLAP`
- `PIPELINE3_TECHNICAL_MAX_DOC_CHARS`

#### Infra
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- `GCS_BUCKET_NAME`
- `GOOGLE_CLOUD_PROJECT` aliases
- `MLFLOW_TRACKING_URI` (runtime env, used by MLflow client)

---

### 8.3 Fixed by code (unless source changes)

- Per-run Neo4j DB random naming strategy and sanitization policy
- Run-stage ordering and orchestration sequence
- Artifact file names logged to MLflow
- Prompt body in eval stage requiring tool DB argument usage
- Node/edge labels and persistence patterns in tools2 (`:Entity`, `:REL`, `r.type`)
- Structural/technical instruction templates in pipeline3 (currently hardcoded constants)
- `kg_track` implementation as boolean flags (`kg_track_structural`, `kg_track_technical`) instead of single string field

---

## 9. Resilience and Failure Semantics

- **DB create failure:** warning only; downstream may still work.
- **pipeline3 partial issues:** metrics logged from whatever payload is available.
- **Neo4j metadata query failure:** captured in `neo4j_count_error`; run continues.
- **Eval failure:** error artifact logged; scoring fetch still attempted.
- **Scoring fetch failure:** logged param with error string.

This design favors observability over strict fail-fast behavior.

---

## 10. Observability Surfaces

### MLflow params
- user/db identity, model config, transform statuses, error flags

### MLflow metrics
- file and graph write counts
- relation-type and entity-kind distributions
- evaluation scores

### MLflow artifacts
- full pipeline result JSON
- graph schema counts JSON
- eval text/error
- scoring record JSON

---

## 11. Practical Tuning Playbook

- If summarization is slow: increase `SUMMARY_MAX_PARALLEL` cautiously.
- If cross-file relationships are weak: increase structural batch overlap.
- If extraction quality is noisy: tighten `ALLOWED_NODE_TYPES`/`ALLOWED_REL_TYPES`.
- If cost is high: reduce `*_MAX_DOC_CHARS`, adjust batch size/overlap, use lower-cost model.
- If consistency matters: pin both summarizer and graph extractor model names explicitly in env.

---

## 12. Known Architectural Nuances

- `pipeline3` docstring mentions `kg_track: structural/technical`, but actual persisted properties are boolean flags.
- Querying `n.kg_track` will often yield null unless code is changed.
- MERGE + `SET +=` can cause one node to carry both track flags (expected for shared entities).
- `Config` snapshot happens at import time; env should be set before running the process.

---

## 13. Appendix: Key Reference Locations

- `Ops_Project/experiment.py`
- `Ops_Project/KG_agent/config.py`
- `Ops_Project/KG_agent/pipeline3.py`
- `Ops_Project/KG_agent/pipeline2.py`
- `Ops_Project/KG_agent/tools2.py`
