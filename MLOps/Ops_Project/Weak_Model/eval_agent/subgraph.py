"""
k-hop neighborhood fetch + serialization (aligned with KG_agent/kg_reconstruct_test.py).

Uses :Entity nodes and :REL relationships as produced by tools2.py.
"""
from __future__ import annotations

from typing import Any

from neo4j import Driver


def fetch_subgraph_k_hops(
    driver: Driver,
    *,
    needle: str | None,
    exact_id: str | None,
    k: int,
    max_starts: int,
    database: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (nodes, edges) for all :Entity nodes reachable within k hops on :REL
    from any start Entity matching needle or exact_id.

    Matching on ``Entity.id`` is **case-insensitive** (exact and substring).
    """
    k = max(1, min(int(k), 10))
    max_starts = max(1, int(max_starts))

    if exact_id is None and not needle:
        raise ValueError("Provide needle or exact_id")

    _sk: dict[str, Any] = {}
    if database:
        _sk["database"] = database

    cypher_nodes = f"""
    MATCH (start:Entity)
    WHERE ($exact_id IS NOT NULL AND toLower(toString(start.id)) = toLower(toString($exact_id)))
       OR ($exact_id IS NULL AND toLower(toString(start.id)) CONTAINS toLower(toString($needle)))
    WITH start LIMIT $max_starts
    MATCH (start)-[:REL*0..{k}]-(n:Entity)
    WITH collect(DISTINCT start) + collect(DISTINCT n) AS bag
    UNWIND bag AS node
    RETURN collect(DISTINCT node) AS nodes
    """

    with driver.session(**_sk) as session:
        row = session.run(
            cypher_nodes,
            exact_id=exact_id,
            needle=needle if needle is not None else "",
            max_starts=max_starts,
        ).single()
        nodes_raw = row["nodes"] if row else []

    node_ids: list[str] = []
    nodes_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in nodes_raw:
        nid = str(node.get("id", node.element_id))
        if nid in seen:
            continue
        seen.add(nid)
        props = dict(node)
        kind = props.get("kind", "")
        node_ids.append(nid)
        nodes_out.append(
            {
                "id": nid,
                "kind": kind,
                "labels": list(node.labels),
                "properties": {key: val for key, val in props.items() if key != "id"},
            }
        )

    edges_out: list[dict[str, Any]] = []
    if not node_ids:
        return [], []

    with driver.session(**_sk) as session:
        er = session.run(
            """
            MATCH (a:Entity)-[r:REL]->(b:Entity)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id AS src, b.id AS dst, r.type AS rel_type, properties(r) AS rel_props
            """,
            ids=node_ids,
        )
        for record in er:
            edges_out.append(
                {
                    "src": record["src"],
                    "dst": record["dst"],
                    "type": record["rel_type"],
                    "properties": record["rel_props"] or {},
                }
            )

    return nodes_out, edges_out


def serialize_subgraph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    title: str,
) -> str:
    excluded_props = {
        "embedding",
        "embedding_text",
        "embedding_text_hash",
        "embedding_model",
    }
    lines = [
        f"## Subgraph: {title}",
        f"### Nodes ({len(nodes)})",
    ]
    for n in nodes:
        raw_extra = n.get("properties") or {}
        # Embedding vectors and embedding-helper fields can explode context size;
        # keep retrieval metadata out of the serialized QA context.
        extra = {k: v for k, v in raw_extra.items() if k not in excluded_props}
        brief = ", ".join(f"{k}={v!r}" for k, v in list(extra.items())[:8])
        if len(extra) > 8:
            brief += ", ..."
        lines.append(f"- **{n['id']}** (kind={n.get('kind')!r}) {brief}")

    lines.append(f"### Edges ({len(edges)})")
    for e in edges:
        rp = e.get("properties") or {}
        t = e.get("type") or "RELATED_TO"
        lines.append(f"- `{e['src']}` --[{t}]--> `{e['dst']}` {rp if rp else ''}")

    return "\n".join(lines)
