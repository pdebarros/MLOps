"""
Collect Neo4j KG statistics for manual evaluation: run Cypher summaries and write one CSV.

Uses the same :Entity / :REL schema as KG_agent (tools2 / pipeline3). Loads credentials
from KG_agent config (.env).

Run from Ops_Project:

  python collect_metadata.py --database mygraph
  python collect_metadata.py --database mygraph --user-id u_123
  python collect_metadata.py --database mygraph --output ./reports/meta.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
_KG = _ROOT / "KG_agent"
if str(_KG) not in sys.path:
    sys.path.insert(0, str(_KG))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline2 import normalize_user_id  # noqa: E402


def _collect_rows(
    database: str,
    user_id: str | None,
    allowed_kinds: list[str],
    allowed_rels: list[str],
) -> list[list[str]]:
    """Return table rows [section, name, value, detail1, detail2]."""
    from config import Config
    from neo4j import GraphDatabase

    prefix: str | None = None
    if user_id is not None:
        uid = normalize_user_id(user_id)
        prefix = f"{uid}/"

    rows: list[list[str]] = []

    def add_summary(key: str, val: Any, d1: str = "", d2: str = "") -> None:
        rows.append(["summary", key, str(val), d1, d2])

    driver = GraphDatabase.driver(
        Config.NEO4J_URI,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )
    try:
        with driver.session(database=database) as session:
            # --- scope line ---
            add_summary("neo4j_database", database)
            add_summary(
                "id_filter",
                f"Entity.id STARTS WITH {prefix!r}" if prefix else "(entire database)",
            )
            if prefix is not None:
                params: dict[str, Any] = {"p": prefix}
            else:
                params = {}

            def entity_where(alias: str = "n") -> str:
                if prefix is None:
                    return ""
                return f"WHERE {alias}.id STARTS WITH $p"

            def rel_both_ends_where() -> str:
                if prefix is None:
                    return ""
                return "WHERE a.id STARTS WITH $p AND b.id STARTS WITH $p"

            # Totals
            q_nodes = f"MATCH (n:Entity) {entity_where()} RETURN count(n) AS c"
            q_edges = f"""
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                {rel_both_ends_where()}
                RETURN count(r) AS c
            """
            nc = session.run(q_nodes, **params).single()
            ec = session.run(q_edges, **params).single()
            add_summary("entity_nodes", int(nc["c"]) if nc else 0)
            add_summary("rel_edges_directed", int(ec["c"]) if ec else 0)

            # Relationship types
            q_rt = f"""
                MATCH (a:Entity)-[r:REL]->(b:Entity)
                {rel_both_ends_where()}
                RETURN coalesce(r.type, '(null)') AS rel_type, count(*) AS c
                ORDER BY c DESC
            """
            for rec in session.run(q_rt, **params):
                rows.append(
                    [
                        "relationship_type",
                        str(rec["rel_type"]),
                        str(int(rec["c"])),
                        "",
                        "",
                    ]
                )

            # Entity kinds
            q_k = f"""
                MATCH (n:Entity)
                {entity_where()}
                RETURN coalesce(n.kind, '(null)') AS kind, count(*) AS c
                ORDER BY c DESC
            """
            for rec in session.run(q_k, **params):
                rows.append(
                    ["entity_kind", str(rec["kind"]), str(int(rec["c"])), "", ""]
                )

            # kg_track (if present)
            q_tr = f"""
                MATCH (n:Entity)
                {entity_where()}
                RETURN coalesce(n.kg_track, '(null)') AS track, count(*) AS c
                ORDER BY c DESC
            """
            for rec in session.run(q_tr, **params):
                rows.append(
                    ["kg_track", str(rec["track"]), str(int(rec["c"])), "", ""]
                )

            # Isolated entities (no REL)
            if prefix is not None:
                q_iso = """
                    MATCH (n:Entity)
                    WHERE n.id STARTS WITH $p AND NOT (n)-[:REL]-()
                    RETURN count(n) AS c
                """
            else:
                q_iso = """
                    MATCH (n:Entity)
                    WHERE NOT (n)-[:REL]-()
                    RETURN count(n) AS c
                """
            irec = session.run(q_iso, **params).single()
            add_summary("isolated_entity_count", int(irec["c"]) if irec else 0)

            # Schema validation vs config ALLOWED_*
            allowed_k_set = set(allowed_kinds)
            allowed_r_set = set(allowed_rels)
            for rec in session.run(q_rt, **params):
                rt = str(rec["rel_type"])
                if rt == "(null)" or rt in allowed_r_set:
                    continue
                rows.append(
                    [
                        "unexpected_rel_type",
                        rt,
                        str(int(rec["c"])),
                        "not_in_ALLOWED_REL_TYPES",
                        "",
                    ]
                )
            for rec in session.run(q_k, **params):
                k = str(rec["kind"])
                if k == "(null)" or k in allowed_k_set:
                    continue
                rows.append(
                    [
                        "unexpected_entity_kind",
                        k,
                        str(int(rec["c"])),
                        "not_in_ALLOWED_NODE_TYPES",
                        "",
                    ]
                )

            # Legacy schema (tools.py): CodeEntity nodes, RELATION edges
            try:
                c = session.run(
                    "MATCH (n:CodeEntity) RETURN count(n) AS c"
                ).single()
                add_summary("legacy_code_entity_nodes", int(c["c"]) if c else 0)
            except Exception:
                add_summary("legacy_code_entity_nodes", "n/a")
            try:
                c = session.run(
                    "MATCH ()-[r:RELATION]->() RETURN count(r) AS c"
                ).single()
                add_summary("legacy_relation_edges", int(c["c"]) if c else 0)
            except Exception:
                add_summary("legacy_relation_edges", "n/a")

            # Isolated entity samples (up to 50)
            if prefix is not None:
                q_samp = """
                    MATCH (n:Entity)
                    WHERE n.id STARTS WITH $p AND NOT (n)-[:REL]-()
                    RETURN n.id AS id, coalesce(n.kind,'') AS kind,
                           coalesce(n.kg_track,'') AS track
                    LIMIT 50
                """
            else:
                q_samp = """
                    MATCH (n:Entity)
                    WHERE NOT (n)-[:REL]-()
                    RETURN n.id AS id, coalesce(n.kind,'') AS kind,
                           coalesce(n.kg_track,'') AS track
                    LIMIT 50
                """
            for rec in session.run(q_samp, **params):
                rows.append(
                    [
                        "sample_isolated_entity",
                        str(rec["id"]),
                        str(rec["kind"]),
                        str(rec["track"]),
                        "",
                    ]
                )

            # Top 30 by incident REL count (Neo4j 5+ disallows size(pattern); use MATCH + count)
            if prefix is not None:
                q_deg = """
                    MATCH (n:Entity)
                    WHERE n.id STARTS WITH $p
                    OPTIONAL MATCH (n)-[r:REL]-()
                    WITH n, count(r) AS deg
                    RETURN n.id AS id, coalesce(n.kind,'') AS kind,
                           coalesce(n.kg_track,'') AS track, deg
                    ORDER BY deg DESC
                    LIMIT 30
                """
            else:
                q_deg = """
                    MATCH (n:Entity)
                    OPTIONAL MATCH (n)-[r:REL]-()
                    WITH n, count(r) AS deg
                    RETURN n.id AS id, coalesce(n.kind,'') AS kind,
                           coalesce(n.kg_track,'') AS track, deg
                    ORDER BY deg DESC
                    LIMIT 30
                """
            for rec in session.run(q_deg, **params):
                rows.append(
                    [
                        "sample_high_degree",
                        str(rec["id"]),
                        str(rec["kind"]),
                        str(rec["track"]),
                        str(int(rec["deg"])),
                    ]
                )

    finally:
        driver.close()

    return rows


def main() -> None:
    from config import Config

    p = argparse.ArgumentParser(
        description="Export Neo4j KG metadata to a CSV for manual inspection."
    )
    p.add_argument(
        "--database",
        required=True,
        help="Neo4j logical database name (same as pipeline --neo4j-database).",
    )
    p.add_argument(
        "--user-id",
        default=None,
        help="If set, only Entity ids starting with <normalized>/ are included.",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output CSV path (default: kg_metadata_<database>.csv in cwd).",
    )
    args = p.parse_args()

    safe_db = "".join(c if c.isalnum() or c in "-_" else "_" for c in args.database)
    out_path = Path(
        args.output if args.output else f"kg_metadata_{safe_db}.csv"
    )

    allowed_kinds = list(Config.ALLOWED_NODE_TYPES)
    allowed_rels = list(Config.ALLOWED_REL_TYPES)

    rows = _collect_rows(
        args.database,
        args.user_id,
        allowed_kinds,
        allowed_rels,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["section", "name_or_type", "value_or_count", "detail1", "detail2"]
        )
        w.writerows(rows)

    print(f"Wrote {len(rows) + 1} lines (incl. header) to {out_path.resolve()}")


if __name__ == "__main__":
    main()
