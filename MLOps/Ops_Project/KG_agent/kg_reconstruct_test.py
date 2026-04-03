"""
Test harness: load a k-hop neighborhood around file-related Entity nodes in Neo4j,
serialize the subgraph as text, and ask Vertex Gemini (ADC) to summarize it.

Usage (from repo root, with .env / ADC configured). Prefer running the file directly
so `KG_agent/__init__.py` is not loaded:

  python KG_agent/kg_reconstruct_test.py --needle pipeline.py --k 2
  python KG_agent/kg_reconstruct_test.py --exact-id "aggregation/pipeline.py" --k 3 --dry-run

Environment: same Neo4j and Vertex settings as tools2.py (NEO4J_*, GOOGLE_CLOUD_PROJECT, etc.).
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from config import Config
from neo4j import GraphDatabase

SUMMARY_PROMPT = """Write a short summary of what this codebase slice represents; only use facts supported by the graph; mark uncertainty."""


def _neo4j_driver():
    return GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )


def _vertex_llm():
    from langchain_google_vertexai import ChatVertexAI

    kwargs: dict[str, Any] = {
        "model": Config.VERTEX_GEMINI_MODEL,
        "location": Config.VERTEX_LOCATION,
        "temperature": Config.VERTEX_TEMPERATURE,
    }
    if Config.GOOGLE_CLOUD_PROJECT:
        kwargs["project"] = Config.GOOGLE_CLOUD_PROJECT
    if Config.VERTEX_MAX_TOKENS is not None:
        kwargs["max_tokens"] = Config.VERTEX_MAX_TOKENS
    return ChatVertexAI(**kwargs)


def fetch_subgraph_k_hops(
    driver,
    *,
    needle: str | None,
    exact_id: str | None,
    k: int,
    max_starts: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (nodes, edges) for all :Entity nodes reachable within k hops on :REL
    from any start Entity matching needle or exact_id.
    """
    k = max(1, min(int(k), 10))
    max_starts = max(1, int(max_starts))

    if exact_id is None and not needle:
        raise ValueError("Provide --needle or --exact-id")

    # *0..k includes the start node (0 hops) so isolated entities still appear.
    cypher_nodes = f"""
    MATCH (start:Entity)
    WHERE ($exact_id IS NOT NULL AND start.id = $exact_id)
       OR ($exact_id IS NULL AND start.id CONTAINS $needle)
    WITH start LIMIT $max_starts
    MATCH (start)-[:REL*0..{k}]-(n:Entity)
    WITH collect(DISTINCT start) + collect(DISTINCT n) AS bag
    UNWIND bag AS node
    RETURN collect(DISTINCT node) AS nodes
    """

    with driver.session() as session:
        row = session.run(
            cypher_nodes,
            exact_id=exact_id,
            needle=needle if needle is not None else "",
            max_starts=max_starts,
        ).single()
        nodes_raw = row["nodes"] if row else []

    node_ids = []
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
                "properties": {k: v for k, v in props.items() if k != "id"},
            }
        )

    edges_out: list[dict[str, Any]] = []
    if not node_ids:
        return [], []

    with driver.session() as session:
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
    lines = [
        f"## Subgraph: {title}",
        f"### Nodes ({len(nodes)})",
    ]
    for n in nodes:
        extra = n.get("properties") or {}
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


def run_summary(serialized_graph: str, llm) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(
            content="You are a careful assistant. When uncertain, say so explicitly."
        ),
        HumanMessage(
            content=f"{SUMMARY_PROMPT}\n\n--- Graph data ---\n\n{serialized_graph}"
        ),
    ]
    resp = llm.invoke(messages)
    return resp.content if hasattr(resp, "content") else str(resp)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Reconstruct-style summary from Neo4j subgraph + Gemini")
    p.add_argument("--needle", default=None, help="Substring match against Entity.id")
    p.add_argument("--exact-id", default=None, dest="exact_id", help="Exact Entity.id")
    p.add_argument("--k", type=int, default=2, help="Hop depth (1-10)")
    p.add_argument(
        "--max-starts",
        type=int,
        default=3,
        dest="max_starts",
        help="Max start nodes when matching by needle",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print serialized graph only; do not call Vertex",
    )
    args = p.parse_args(argv)

    if not args.exact_id and not args.needle:
        print("Error: provide --needle or --exact-id", file=sys.stderr)
        return 2

    driver = _neo4j_driver()
    try:
        nodes, edges = fetch_subgraph_k_hops(
            driver,
            needle=args.needle,
            exact_id=args.exact_id,
            k=args.k,
            max_starts=args.max_starts,
        )
    finally:
        driver.close()

    title = args.exact_id or args.needle or "unknown"
    text = serialize_subgraph(nodes, edges, title=title)

    print(text)
    print()

    if args.dry_run:
        return 0

    if not nodes:
        print("No nodes in subgraph; skipping LLM call.", file=sys.stderr)
        return 1

    llm = _vertex_llm()
    summary = run_summary(text, llm)
    print("### Model summary")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
