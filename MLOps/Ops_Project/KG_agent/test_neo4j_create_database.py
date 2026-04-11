#!/usr/bin/env python3
"""
Smoke test: verify this Neo4j Bolt user can create a **new logical database**
(multi-database feature). Uses the same env as ``config.py`` (``KG_agent/.env``).

Run from ``KG_agent``:

  cd Ops_Project/KG_agent
  python test_neo4j_create_database.py

Or from ``Ops_Project``:

  python KG_agent/test_neo4j_create_database.py

Exit code: 0 if create + verify succeeded, 1 otherwise.
"""
from __future__ import annotations

import argparse
import re
import sys
import uuid
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import Config  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402


def _sanitize_db_name(raw: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "").strip())
    if not s:
        s = "testdb"
    if not s[0].isalpha():
        s = "t_" + s
    return s[:48]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a throwaway Neo4j database and verify it is usable."
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Logical database name (default: test_db_<8 hex chars>).",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="After a successful test, run DROP DATABASE for the test name.",
    )
    args = parser.parse_args()

    db_name = _sanitize_db_name(args.name or f"test_db_{uuid.uuid4().hex[:8]}")
    uri = Config.NEO4J_URI
    user = Config.NEO4J_USER

    print(f"Neo4j URI: {uri}")
    print(f"Neo4j user: {user}")
    print(f"Test database name: {db_name}")
    print()

    driver = GraphDatabase.driver(
        uri,
        auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD),
    )
    try:
        # Server / edition info (Neo4j 4+)
        try:
            with driver.session(database="system") as session:
                rows = session.run(
                    "CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition"
                )
                for r in rows:
                    print(
                        f"Component: {r['name']}  versions={r['versions']}  edition={r.get('edition')}"
                    )
        except Exception as e:
            print(f"(dbms.components unavailable: {e})")

        print()
        print("Step 1: CREATE DATABASE ... IF NOT EXISTS (system session)")
        try:
            with driver.session(database="system") as session:
                session.run(f"CREATE DATABASE `{db_name}` IF NOT EXISTS")
        except Exception as e:
            print(f"FAILED: {e}")
            return 1

        print("Step 2: SHOW DATABASES (check name appears)")
        try:
            with driver.session(database="system") as session:
                rows = list(session.run("SHOW DATABASES"))
            names = [str(r.get("name")) for r in rows if r.get("name") is not None]
            if db_name not in names:
                print(
                    f"FAILED: database {db_name!r} not in SHOW DATABASES after CREATE. "
                    f"Known names (sample): {names[:20]}{'...' if len(names) > 20 else ''}"
                )
                return 1
            meta = next((dict(r) for r in rows if str(r.get("name")) == db_name), {})
            print(f"OK: listed — {meta}")
        except Exception as e:
            print(f"FAILED (SHOW DATABASES): {e}")
            return 1

        print("Step 3: Open session to new database and run RETURN 1")
        try:
            with driver.session(database=db_name) as session:
                one = session.run("RETURN 1 AS ok").single()
                assert one and one["ok"] == 1
        except Exception as e:
            print(f"FAILED: {e}")
            return 1
        print("OK: query succeeded on new database.")

        if args.drop:
            print()
            print(f"Step 4: DROP DATABASE `{db_name}` IF EXISTS (destructive)")
            try:
                with driver.session(database="system") as session:
                    session.run(f"DROP DATABASE `{db_name}` IF EXISTS")
                print("OK: dropped.")
            except Exception as e:
                print(f"FAILED: {e}")
                return 1

        print()
        print("SUCCESS: this instance accepts CREATE DATABASE for this user.")
        return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
