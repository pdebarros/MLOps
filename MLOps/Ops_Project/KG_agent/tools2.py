from __future__ import annotations

import logging
import re
from typing import Any, List

from config import Config
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_experimental.graph_transformers import LLMGraphTransformer
from neo4j import GraphDatabase

logger = logging.getLogger("Orchestrator.KG-Tool2")

# Neo4j connection
NEO4J_URI = Config.NEO4J_URI
NEO4J_USER = Config.NEO4J_USER
NEO4J_PASSWORD = Config.NEO4J_PASSWORD

# Graph LLM: "vertex" (Gemini on Vertex AI via ADC) or "huggingface" (local HF).
KG_GRAPH_BACKEND = Config.KG_GRAPH_BACKEND

# Vertex AI / Gemini (uses Application Default Credentials: gcloud auth application-default login,
# workload identity, or GOOGLE_APPLICATION_CREDENTIALS — same as GCS / ADK.)
VERTEX_GEMINI_MODEL = Config.VERTEX_GEMINI_MODEL
GOOGLE_CLOUD_PROJECT = Config.GOOGLE_CLOUD_PROJECT
VERTEX_LOCATION = Config.VERTEX_LOCATION
VERTEX_TEMPERATURE = Config.VERTEX_TEMPERATURE
VERTEX_MAX_TOKENS = Config.VERTEX_MAX_TOKENS
# If True, force unstructured JSON path (json-repair) even on Vertex — rarely needed.
KG_VERTEX_IGNORE_TOOL_USAGE = Config.KG_VERTEX_IGNORE_TOOL_USAGE

# Hugging Face model settings
HF_GRAPH_MODEL = Config.HF_GRAPH_MODEL
HF_MAX_NEW_TOKENS = Config.HF_MAX_NEW_TOKENS
HF_TEMPERATURE = Config.HF_TEMPERATURE
# Hard cap input text size to avoid long-sequence crashes and lower RAM.
KG_MAX_DOC_CHARS = Config.KG_MAX_DOC_CHARS

# Global-edge batching settings
KG_BATCH_SIZE = Config.KG_BATCH_SIZE
KG_BATCH_OVERLAP = Config.KG_BATCH_OVERLAP


def _estimate_tokens_approx(text: str | None) -> int:
    """Lightweight token estimate (~4 chars/token) for cost trending."""
    if not text:
        return 0
    s = str(text)
    return max(1, (len(s) + 3) // 4)


def _graph_model_name_for_backend() -> str:
    if KG_GRAPH_BACKEND in ("huggingface", "hf", "local"):
        return HF_GRAPH_MODEL
    return VERTEX_GEMINI_MODEL

# Optional schema constraints to stabilize Neo4j structure
ALLOWED_NODE_TYPES = list(Config.ALLOWED_NODE_TYPES)
ALLOWED_REL_TYPES = list(Config.ALLOWED_REL_TYPES)

_hf_llm = None
_vertex_llm = None
_graph_transformer: LLMGraphTransformer | None = None
_neo4j_driver = None


def _sanitize_tenant_label(uid: str) -> str:
    """
    Convert a user_id into a valid Neo4j label segment.

    Rules:
      - Replace any character outside [A-Za-z0-9_] with '_'
      - Prefix with 'U' when the result starts with a digit
      - Prepend 'User_' so every tenant label has the form  User_<safe_uid>

    Example: 'u_123' → 'User_u_123',  '99foo' → 'User_U99foo'
    """
    safe = re.sub(r"[^A-Za-z0-9_]", "_", uid)
    if safe and safe[0].isdigit():
        safe = "U" + safe
    return f"User_{safe}"


def _get_vertex_llm() -> BaseLanguageModel:
    """Gemini on Vertex AI using Application Default Credentials (no API key in code)."""
    global _vertex_llm
    if _vertex_llm is not None:
        return _vertex_llm

    from langchain_google_vertexai import ChatVertexAI

    kwargs: dict[str, Any] = {
        "model": VERTEX_GEMINI_MODEL,
        "location": VERTEX_LOCATION,
        "temperature": VERTEX_TEMPERATURE,
    }
    if GOOGLE_CLOUD_PROJECT:
        kwargs["project"] = GOOGLE_CLOUD_PROJECT
    if VERTEX_MAX_TOKENS is not None and str(VERTEX_MAX_TOKENS).strip() != "":
        kwargs["max_tokens"] = int(VERTEX_MAX_TOKENS)

    logger.info(
        "Using Vertex AI Gemini for graph extraction (model=%s, location=%s, project=%s)",
        VERTEX_GEMINI_MODEL,
        VERTEX_LOCATION,
        GOOGLE_CLOUD_PROJECT or "(default from ADC)",
    )
    _vertex_llm = ChatVertexAI(**kwargs)
    return _vertex_llm


def _get_hf_llm() -> BaseLanguageModel:
    """Create the Hugging Face LLM lazily."""
    global _hf_llm
    if _hf_llm is not None:
        return _hf_llm

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
    from langchain_huggingface import HuggingFacePipeline

    # Build raw transformers pipeline so we can use device="mps" on Apple Silicon.
    if torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
        device_name = "mps"
    elif torch.cuda.is_available():
        device = 0
        torch_dtype = torch.float16
        device_name = "cuda"
    else:
        device = -1
        torch_dtype = torch.float32
        device_name = "cpu"

    logger.info("Using HF device: %s", device_name)

    pipeline_kwargs = {
        "max_new_tokens": HF_MAX_NEW_TOKENS,
        # Return only the completion text for faster downstream parsing.
        "return_full_text": False,
    }
    if HF_TEMPERATURE > 0:
        pipeline_kwargs["temperature"] = HF_TEMPERATURE
        pipeline_kwargs["do_sample"] = True
    else:
        # Greedy decoding when temperature is zero.
        pipeline_kwargs["do_sample"] = False

    tokenizer = AutoTokenizer.from_pretrained(HF_GRAPH_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        HF_GRAPH_MODEL,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    text_gen_pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        **pipeline_kwargs,
    )
    _hf_llm = HuggingFacePipeline(
        pipeline=text_gen_pipe,
        model_id=HF_GRAPH_MODEL,
        pipeline_kwargs=pipeline_kwargs,
    )
    return _hf_llm


def _get_graph_llm() -> BaseLanguageModel:
    if KG_GRAPH_BACKEND in ("huggingface", "hf", "local"):
        return _get_hf_llm()
    if KG_GRAPH_BACKEND in ("vertex", "gemini", "gcp"):
        return _get_vertex_llm()
    raise ValueError(
        f"Unknown KG_GRAPH_BACKEND={KG_GRAPH_BACKEND!r}; use 'vertex' or 'huggingface'."
    )


def _get_graph_transformer() -> LLMGraphTransformer:
    """Create a schema-constrained graph transformer."""
    global _graph_transformer
    if _graph_transformer is not None:
        return _graph_transformer

    llm = _get_graph_llm()
    # Vertex Gemini supports structured output / tool-style extraction; HF pipelines do not.
    use_hf = KG_GRAPH_BACKEND in ("huggingface", "hf", "local")
    ignore_tools = use_hf or KG_VERTEX_IGNORE_TOOL_USAGE
    _graph_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODE_TYPES,
        allowed_relationships=ALLOWED_REL_TYPES,
        strict_mode=False,
        ignore_tool_usage=ignore_tools,
        additional_instructions=(
            "Use stable, canonical entity names. "
            "Prefer uppercase relation labels from the allowed list."
        ),
    )
    return _graph_transformer


def _get_neo4j_driver():
    """Create Neo4j driver lazily (no APOC dependency)."""
    global _neo4j_driver
    if _neo4j_driver is not None:
        return _neo4j_driver

    _neo4j_driver = GraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
    return _neo4j_driver


def _persist_graph_documents(
    graph_documents: list,
    *,
    extra_node_props: dict[str, Any] | None = None,
    database: str | None = None,
    tenant_id: str | None = None,
) -> tuple[int, int]:
    """
    Persist LangChain GraphDocuments to Neo4j without APOC.

    Multi-tenancy (AuraDB / single-database):
      When ``tenant_id`` is provided, two isolation mechanisms are applied:

      1. **Label-Based** — each node gets an extra ``:`User_<uid>`` label so
         Neo4j index scans jump directly to that tenant's sub-graph without
         touching other tenants' nodes.

      2. **Property-Based** — every node and relationship carries
         ``tenantId = <uid>`` so GraphRAG Cypher filters stay unambiguous.

      MERGE keys include ``tenantId``, ensuring that two tenants whose files
      share the same entity name are stored as distinct nodes.

    ``database`` — Neo4j 4+ logical database name; ``None`` uses server default.
    """
    driver = _get_neo4j_driver()
    nodes_written = 0
    edges_written = 0
    session_kwargs: dict[str, Any] = {}
    if database:
        session_kwargs["database"] = database

    # _sanitize_tenant_label produces only [A-Za-z0-9_] so backtick injection
    # is not possible; the f-string dynamic label is therefore safe.
    user_label = _sanitize_tenant_label(tenant_id) if tenant_id else None

    with driver.session(**session_kwargs) as session:
        for gd in graph_documents:
            for node in gd.nodes:
                node_id = str(node.id)
                node_type = str(node.type or "Entity")
                props = dict(node.properties or {})
                props["kind"] = node_type
                merged_props = dict(props)
                if extra_node_props:
                    merged_props.update(extra_node_props)

                if tenant_id:
                    merged_props["tenantId"] = tenant_id
                    session.run(
                        f"""
                        MERGE (n:Entity:`{user_label}` {{id: $id, tenantId: $tenant_id}})
                        SET n += $props
                        """,
                        id=node_id,
                        tenant_id=tenant_id,
                        props=merged_props,
                    )
                else:
                    session.run(
                        """
                        MERGE (n:Entity {id: $id})
                        SET n += $props
                        """,
                        id=node_id,
                        props=merged_props,
                    )
                nodes_written += 1

            for rel in gd.relationships:
                src_id = str(rel.source.id)
                dst_id = str(rel.target.id)
                rel_type = str(rel.type or "RELATED_TO")
                rel_props = dict(rel.properties or {})
                rel_props["type"] = rel_type

                if tenant_id:
                    rel_props["tenantId"] = tenant_id
                    session.run(
                        f"""
                        MATCH (a:Entity:`{user_label}` {{id: $src_id, tenantId: $tenant_id}})
                        MATCH (b:Entity:`{user_label}` {{id: $dst_id, tenantId: $tenant_id}})
                        MERGE (a)-[r:REL {{type: $rel_type, tenantId: $tenant_id}}]->(b)
                        SET r += $rel_props
                        """,
                        src_id=src_id,
                        dst_id=dst_id,
                        rel_type=rel_type,
                        tenant_id=tenant_id,
                        rel_props=rel_props,
                    )
                else:
                    session.run(
                        """
                        MATCH (a:Entity {id: $src_id})
                        MATCH (b:Entity {id: $dst_id})
                        MERGE (a)-[r:REL {type: $rel_type}]->(b)
                        SET r += $rel_props
                        """,
                        src_id=src_id,
                        dst_id=dst_id,
                        rel_type=rel_type,
                        rel_props=rel_props,
                    )
                edges_written += 1
    return nodes_written, edges_written


def _pick_completed_docs(
    results: list,
    max_doc_chars: int | None = None,
) -> List[Document]:
    cap = KG_MAX_DOC_CHARS if max_doc_chars is None else max_doc_chars
    docs: List[Document] = []
    for r in results:
        if r.get("status") == "completed" and r.get("summary"):
            file_name = r.get("file") or "unknown_file"
            summary = str(r["summary"])
            if cap > 0 and len(summary) > cap:
                summary = summary[:cap]
            docs.append(
                Document(
                    page_content=f"FILE: {file_name}\nSUMMARY:\n{summary}",
                    metadata={"file": file_name},
                )
            )
    return docs


def _chunk_docs(docs: List[Document], size: int, overlap: int) -> List[List[Document]]:
    if size <= 0:
        size = 10
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = size - 1

    chunks: List[List[Document]] = []
    start = 0
    step = size - overlap
    while start < len(docs):
        chunks.append(docs[start : start + size])
        start += step
    return chunks


def _to_batch_documents(chunks: List[List[Document]]) -> List[Document]:
    """
    Build one larger Document per chunk so transformer can infer cross-file edges.
    """
    batch_docs: List[Document] = []
    for idx, chunk in enumerate(chunks):
        merged = "\n\n---\n\n".join(d.page_content for d in chunk)
        files = [d.metadata.get("file") for d in chunk]
        batch_docs.append(
            Document(
                page_content=merged,
                metadata={"batch": idx, "files": files},
            )
        )
    return batch_docs


def build_kg_and_push_to_neo4j(
    results: list,
    *,
    kg_batch_size: int | None = None,
    kg_batch_overlap: int | None = None,
    kg_max_doc_chars: int | None = None,
    extra_entity_props: dict[str, Any] | None = None,
    neo4j_database: str | None = None,
    tenant_id: str | None = None,
) -> dict[str, Any]:
    """
    Convert code summaries into GraphDocuments via LLMGraphTransformer (Vertex Gemini by default,
    or local Hugging Face if KG_GRAPH_BACKEND=huggingface), then persist into Neo4j.

    Optional overrides:
      kg_batch_size / kg_batch_overlap — multi-file batching for cross-file edges
      kg_max_doc_chars — per-summary cap when building Documents
      extra_entity_props — merged onto each :Entity (e.g. {\"kg_track\": \"structural\"})
      neo4j_database — Neo4j 4+ database name (default server DB if omitted)
      tenant_id — when set, enables AuraDB logical multi-tenancy via label + property
                  isolation (see _persist_graph_documents for details)
    """
    bs = KG_BATCH_SIZE if kg_batch_size is None else kg_batch_size
    bo = KG_BATCH_OVERLAP if kg_batch_overlap is None else kg_batch_overlap

    docs = _pick_completed_docs(results, max_doc_chars=kg_max_doc_chars)
    if not docs:
        return {"status": "error", "message": "No successful summaries found."}

    chunks = _chunk_docs(docs, bs, bo)
    batch_docs = _to_batch_documents(chunks)
    input_tokens_est = sum(_estimate_tokens_approx(d.page_content) for d in batch_docs)

    transformer = _get_graph_transformer()

    graph_documents = []
    for i, batch_doc in enumerate(batch_docs, start=1):
        logger.info(
            "Transforming chunk %d/%d (%d file summaries)...",
            i,
            len(batch_docs),
            len(batch_doc.metadata.get("files", [])),
        )
        try:
            out = transformer.convert_to_graph_documents([batch_doc])
            graph_documents.extend(out)
        except Exception as e:
            logger.error("Chunk %d failed during graph transform: %s", i, e)

    if not graph_documents:
        return {"status": "error", "message": "No graph documents produced."}

    # Heuristic output-token estimate from extracted graph size.
    out_nodes = sum(len(getattr(gd, "nodes", []) or []) for gd in graph_documents)
    out_edges = sum(len(getattr(gd, "relationships", []) or []) for gd in graph_documents)
    output_tokens_est = max(0, (out_nodes * 12) + (out_edges * 8))

    try:
        nodes_written, edges_written = _persist_graph_documents(
            graph_documents,
            extra_node_props=extra_entity_props,
            database=neo4j_database,
            tenant_id=tenant_id,
        )
    except Exception as e:
        logger.error("Neo4j write failed: %s", e)
        return {"status": "neo4j_error", "message": str(e)}

    return {
        "status": "success",
        "files_processed": len(docs),
        "chunks_processed": len(batch_docs),
        "graph_documents": len(graph_documents),
        "nodes_written": nodes_written,
        "edges_written": edges_written,
        "token_usage": {
            "model": _graph_model_name_for_backend(),
            "backend": KG_GRAPH_BACKEND,
            "method": "estimated_input_chars_div_4_plus_graph_size_heuristic",
            "input_tokens_est": int(input_tokens_est),
            "output_tokens_est": int(output_tokens_est),
            "total_tokens_est": int(input_tokens_est + output_tokens_est),
        },
    }
