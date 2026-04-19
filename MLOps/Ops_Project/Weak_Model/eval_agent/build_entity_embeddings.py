"""
Backfill/update :Entity node embeddings and create a Neo4j vector index.

Designed for Weak_Model/eval_agent GraphRAG retrieval.

Examples (run from Ops_Project):

  python -m Weak_Model.eval_agent.build_entity_embeddings --neo4j-database wtnycf
  python -m Weak_Model.eval_agent.build_entity_embeddings --neo4j-database wtnycf --rebuild-all
  python -m Weak_Model.eval_agent.build_entity_embeddings --neo4j-database wtnycf --index-name my_index
"""
from __future__ import annotations

import argparse
import hashlib
import json
from typing import Any

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from .config import config

DEFAULT_INDEX = config.GEMINI_GRAPHRAG_VECTOR_INDEX
DEFAULT_MODEL = config.GEMINI_GRAPHRAG_EMBED_MODEL
DEFAULT_TOPK = config.GEMINI_GRAPHRAG_TOP_K


def _neo4j_driver():
    return GraphDatabase.driver(
        config.NEO4J_URI,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )


def _coalesce_text(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _entity_text(props: dict[str, Any]) -> str:
    """
    Build deterministic text used for node embedding.

    Keep this mostly identifier/metadata oriented so retrieval selects relevant seeds,
    while actual reasoning still comes from traversing :REL in evaluation.
    """
    id_s = _coalesce_text(props.get("id"))
    kind_s = _coalesce_text(props.get("kind"))
    file_s = _coalesce_text(props.get("file"))
    module_s = _coalesce_text(props.get("module"))
    path_s = _coalesce_text(props.get("path"))
    track_bits: list[str] = []
    if props.get("kg_track_structural") is True:
        track_bits.append("structural")
    if props.get("kg_track_technical") is True:
        track_bits.append("technical")
    track_s = ",".join(track_bits)

    parts = [f"id: {id_s}", f"kind: {kind_s}"]
    if file_s:
        parts.append(f"file: {file_s}")
    if module_s:
        parts.append(f"module: {module_s}")
    if path_s:
        parts.append(f"path: {path_s}")
    if track_s:
        parts.append(f"track: {track_s}")

    # Add a compact "extra" tail for scalar props not already represented.
    ignored = {
        "id",
        "kind",
        "file",
        "module",
        "path",
        "embedding",
        "embedding_text",
        "embedding_text_hash",
        "embedding_model",
        "kg_track_structural",
        "kg_track_technical",
    }
    extras: list[str] = []
    for k in sorted(props.keys()):
        if k in ignored:
            continue
        v = props.get(k)
        if isinstance(v, (str, int, float, bool)):
            s = _coalesce_text(v)
            if s:
                extras.append(f"{k}={s}")
        if len(extras) >= 8:
            break
    if extras:
        parts.append("extra: " + "; ".join(extras))
    return " | ".join(parts)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _vector_dim(embedder: SentenceTransformer) -> int:
    return int(embedder.get_sentence_embedding_dimension())


def _ensure_vector_index(
    session,
    *,
    index_name: str,
    property_name: str,
    dimensions: int,
    similarity: str,
) -> None:
    safe_index = index_name.replace("`", "")
    safe_prop = property_name.replace("`", "")
    safe_sim = similarity.strip().lower()
    if safe_sim not in {"cosine", "euclidean"}:
        raise ValueError("similarity must be one of: cosine, euclidean")

    session.run(
        f"""
        CREATE VECTOR INDEX `{safe_index}` IF NOT EXISTS
        FOR (n:Entity) ON (n.`{safe_prop}`)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: $dims,
          `vector.similarity_function`: $sim
        }}}}
        """,
        dims=dimensions,
        sim=safe_sim,
    )


def _upsert_embeddings(
    session,
    batch: list[dict[str, Any]],
    *,
    property_name: str,
) -> None:
    safe_prop = property_name.replace("`", "")
    session.run(
        f"""
        UNWIND $rows AS row
        MATCH (n) WHERE elementId(n) = row.eid
        SET n.`{safe_prop}` = row.embedding,
            n.embedding_text = row.embedding_text,
            n.embedding_text_hash = row.embedding_text_hash,
            n.embedding_model = row.embedding_model
        """,
        rows=batch,
    )


def _iter_entities(session):
    result = session.run("MATCH (n:Entity) RETURN elementId(n) AS eid, properties(n) AS props")
    for record in result:
        yield str(record["eid"]), dict(record["props"] or {})


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate/update :Entity embeddings and vector index for a Neo4j database."
    )
    p.add_argument("--neo4j-database", required=True, help="Neo4j logical database name.")
    p.add_argument(
        "--index-name",
        default=DEFAULT_INDEX,
        help=f"Vector index name (default: {DEFAULT_INDEX}).",
    )
    p.add_argument(
        "--embedding-property",
        default="embedding",
        help="Node property storing embedding vectors (default: embedding).",
    )
    p.add_argument(
        "--embed-model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--similarity",
        default="cosine",
        choices=["cosine", "euclidean"],
        help="Vector similarity function for Neo4j index (default: cosine).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding/write batch size (default: 128).",
    )
    p.add_argument(
        "--rebuild-all",
        action="store_true",
        help="Embed all entities, ignoring existing hash/model markers.",
    )
    args = p.parse_args()

    db = (args.neo4j_database or "").strip()
    if not db:
        raise ValueError("--neo4j-database must be non-empty.")
    batch_size = max(1, int(args.batch_size))

    embedder = SentenceTransformer(args.embed_model)
    dims = _vector_dim(embedder)
    print(
        f"Embedding model={args.embed_model} dim={dims} db={db} "
        f"index={args.index_name} property={args.embedding_property}"
    )

    driver = _neo4j_driver()
    try:
        with driver.session(database=db) as session:
            _ensure_vector_index(
                session,
                index_name=args.index_name,
                property_name=args.embedding_property,
                dimensions=dims,
                similarity=args.similarity,
            )

            to_encode_texts: list[str] = []
            to_encode_rows: list[dict[str, Any]] = []

            total = 0
            skipped = 0
            written = 0

            for eid, props in _iter_entities(session):
                total += 1
                text = _entity_text(props)
                h = _text_hash(text)
                existing_h = _coalesce_text(props.get("embedding_text_hash"))
                existing_model = _coalesce_text(props.get("embedding_model"))
                has_vec = props.get(args.embedding_property) is not None

                if (
                    not args.rebuild_all
                    and has_vec
                    and existing_h == h
                    and existing_model == args.embed_model
                ):
                    skipped += 1
                    continue

                to_encode_texts.append(text)
                to_encode_rows.append(
                    {
                        "eid": eid,
                        "embedding_text": text,
                        "embedding_text_hash": h,
                        "embedding_model": args.embed_model,
                    }
                )

                if len(to_encode_texts) >= batch_size:
                    vectors = embedder.encode(
                        to_encode_texts,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    )
                    payload: list[dict[str, Any]] = []
                    for row, vec in zip(to_encode_rows, vectors):
                        payload.append(
                            {
                                **row,
                                "embedding": vec.tolist(),
                            }
                        )
                    _upsert_embeddings(
                        session,
                        payload,
                        property_name=args.embedding_property,
                    )
                    written += len(payload)
                    print(f"Embedded + wrote {written} nodes...")
                    to_encode_texts.clear()
                    to_encode_rows.clear()

            if to_encode_texts:
                vectors = embedder.encode(
                    to_encode_texts,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                payload = []
                for row, vec in zip(to_encode_rows, vectors):
                    payload.append({**row, "embedding": vec.tolist()})
                _upsert_embeddings(
                    session,
                    payload,
                    property_name=args.embedding_property,
                )
                written += len(payload)

            summary = {
                "database": db,
                "index_name": args.index_name,
                "embedding_property": args.embedding_property,
                "embedding_model": args.embed_model,
                "vector_dimensions": dims,
                "total_entities_seen": total,
                "entities_skipped_unchanged": skipped,
                "entities_embedded_written": written,
                "rebuild_all": bool(args.rebuild_all),
            }
            print(json.dumps(summary, indent=2))
    finally:
        driver.close()


if __name__ == "__main__":
    main()

