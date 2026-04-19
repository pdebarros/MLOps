"""
Analyze one or more CSVs produced by collect_metadata.py.

Parses the long-format CSV (section, name_or_type, value_or_count, ...) into a
wide table (one row per file), optionally merges configuration labels from a
manifest, and writes summary statistics, correlations, and optional group
comparisons.

Usage (from Ops_Project):

  # All metadata CSVs in a folder (Neo4j database name + node counts per run)
  python analyze_kg_metadata.py --folder ./my_experiment_exports

  # Same, non-recursive (default); use --recursive for subfolders
  python analyze_kg_metadata.py --folder ./reports --output-dir ./reports/analysis

  # Explicit file list (unchanged)
  python analyze_kg_metadata.py --inputs kg_metadata_a.csv kg_metadata_b.csv

  # Many files + manifest with config columns (file column matches basename)
  python analyze_kg_metadata.py \\
    --inputs reports/*.csv \\
    --manifest manifest.csv \\
    --output-dir ./analysis_out

  # Optional plots (requires matplotlib)
  python analyze_kg_metadata.py --inputs reports/*.csv --manifest m.csv --plot

Manifest CSV: include either `file` (basename of each metadata CSV) or `path`
(full path to that CSV). Add any configuration dimensions as extra columns
(kg_model, summarizer, pipeline variant, etc.) and optional numeric scores
(eval_f1, latency_sec) for correlation with graph metrics.

Example manifest.csv::

  file,kg_model,summarizer,eval_f1
  kg_metadata_vertex.csv,vertex,gemini-2.5-flash,0.82
  kg_metadata_hf.csv,huggingface,llama-3.1-8b,0.79
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _safe_int(x: str) -> int | None:
    try:
        return int(x, 10)
    except (TypeError, ValueError):
        return None


def _shannon_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def parse_collect_metadata_csv(path: Path) -> dict[str, Any]:
    """
    Turn a collect_metadata.py CSV into a flat dict of metrics + distributions.
    """
    out: dict[str, Any] = {"source_file": str(path.resolve()), "basename": path.name}
    summary: dict[str, Any] = {}
    rel_types: Counter[str] = Counter()
    entity_kinds: Counter[str] = Counter()
    kg_tracks: Counter[str] = Counter()
    unexpected_rel = 0
    unexpected_kind = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        expected = {
            "section",
            "name_or_type",
            "value_or_count",
            "detail1",
            "detail2",
        }
        if reader.fieldnames is None:
            raise ValueError(f"No header in {path}")
        fields = set(reader.fieldnames)
        if not expected.issubset(fields):
            raise ValueError(
                f"{path}: expected columns {sorted(expected)}, got {reader.fieldnames}"
            )

        for row in reader:
            sec = (row.get("section") or "").strip()
            name = (row.get("name_or_type") or "").strip()
            val = (row.get("value_or_count") or "").strip()

            if sec == "summary":
                vi = _safe_int(val)
                summary[name] = vi if vi is not None else val
            elif sec == "relationship_type":
                rel_types[name] = int(val) if val.isdigit() else 0
            elif sec == "entity_kind":
                entity_kinds[name] = int(val) if val.isdigit() else 0
            elif sec == "kg_track":
                kg_tracks[name] = int(val) if val.isdigit() else 0
            elif sec == "unexpected_rel_type":
                unexpected_rel += int(val) if val.isdigit() else 0
            elif sec == "unexpected_entity_kind":
                unexpected_kind += int(val) if val.isdigit() else 0

    out.update({f"summary__{k}": v for k, v in summary.items()})

    # Canonical column for comparing runs (from collect_metadata summary row)
    out["neo4j_database"] = str(summary.get("neo4j_database", "") or "")

    nodes = _safe_int(str(summary.get("entity_nodes", ""))) or 0
    edges = _safe_int(str(summary.get("rel_edges_directed", ""))) or 0
    isolated = _safe_int(str(summary.get("isolated_entity_count", ""))) or 0

    out["graph__entity_nodes"] = nodes
    out["graph__rel_edges_directed"] = edges
    out["graph__isolated_entity_count"] = isolated
    out["graph__edge_per_node"] = (edges / nodes) if nodes else float("nan")
    out["graph__isolation_rate"] = (isolated / nodes) if nodes else float("nan")
    out["graph__rel_type_count"] = len(rel_types)
    out["graph__entity_kind_count"] = len(entity_kinds)
    out["graph__kg_track_count"] = len(kg_tracks)
    out["graph__rel_type_entropy"] = _shannon_entropy(rel_types)
    out["graph__entity_kind_entropy"] = _shannon_entropy(entity_kinds)
    out["graph__unexpected_rel_total"] = unexpected_rel
    out["graph__unexpected_kind_total"] = unexpected_kind

    # Top relationship type share (concentration)
    if rel_types:
        top = max(rel_types.values())
        out["graph__top_rel_type_share"] = top / sum(rel_types.values())
    else:
        out["graph__top_rel_type_share"] = float("nan")

    out["graph__rel_type_counts_json"] = json.dumps(dict(rel_types), sort_keys=True)
    out["graph__entity_kind_counts_json"] = json.dumps(dict(entity_kinds), sort_keys=True)
    out["graph__kg_track_counts_json"] = json.dumps(dict(kg_tracks), sort_keys=True)

    return out


def discover_metadata_csvs(folder: Path, *, recursive: bool) -> list[Path]:
    """Return sorted paths to *.csv under folder (collect_metadata exports)."""
    root = folder.resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")
    if recursive:
        paths = sorted(root.rglob("*.csv"))
    else:
        paths = sorted(root.glob("*.csv"))
    return [p for p in paths if p.is_file()]


def reorder_wide_columns(df: Any) -> Any:
    """Put database id, file id, and core graph counts first for inspection."""
    import pandas as pd

    priority = [
        "neo4j_database",
        "basename",
        "source_file",
        "graph__entity_nodes",
        "graph__rel_edges_directed",
        "graph__isolated_entity_count",
        "graph__edge_per_node",
        "graph__isolation_rate",
        "graph__rel_type_count",
        "graph__entity_kind_count",
        "graph__kg_track_count",
        "graph__rel_type_entropy",
        "graph__entity_kind_entropy",
        "graph__top_rel_type_share",
        "graph__unexpected_rel_total",
        "graph__unexpected_kind_total",
    ]
    seen: set[str] = set()
    ordered: list[str] = []
    for c in priority:
        if c in df.columns and c not in seen:
            ordered.append(c)
            seen.add(c)
    for c in df.columns:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return df[ordered]


def _load_manifest(manifest_path: Path) -> Any:
    import pandas as pd

    m = pd.read_csv(manifest_path)
    cols = {c.lower() for c in m.columns}
    if "file" not in cols and "path" not in cols:
        raise ValueError(
            "Manifest must contain a `file` (basename) or `path` (full path) column."
        )
    return m


def _merge_manifest_path(rows: list[dict[str, Any]], manifest: Any) -> Any:
    import pandas as pd

    df = pd.DataFrame(rows)
    path_cols = [c for c in manifest.columns if c.lower() == "path"]
    if not path_cols:
        return _merge_manifest_simple(rows, manifest)
    pc = path_cols[0]
    man = manifest.copy()
    man["_resolved"] = man[pc].map(lambda p: str(Path(p).expanduser().resolve()))
    df["_resolved"] = df["source_file"]
    merged = df.merge(
        man,
        left_on="_resolved",
        right_on="_resolved",
        how="left",
        suffixes=("", "_m"),
    )
    merged = merged.drop(columns=["_resolved"], errors="ignore")
    return merged


def _merge_manifest_simple(rows: list[dict[str, Any]], manifest: Any) -> Any:
    import pandas as pd

    df = pd.DataFrame(rows)
    fk = [c for c in manifest.columns if c.lower() == "file"][0]
    man = manifest.copy()
    man["_key"] = man[fk].map(lambda x: Path(str(x)).name)
    df["_key"] = df["basename"]
    merged = df.merge(man, on="_key", how="left", suffixes=("", "_dup"))
    merged = merged.drop(columns=["_key"], errors="ignore")
    # Remove duplicate basename column if present
    if "basename_dup" in merged.columns:
        merged = merged.drop(columns=["basename_dup"])
    return merged


def numeric_columns(df: Any) -> list[str]:
    import pandas as pd

    out: list[str] = []
    for c in df.columns:
        if c.startswith("_"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 2:
            out.append(c)
    return out


def write_summary_by_group(
    df: Any, group_cols: list[str], metric_cols: list[str], out: Path
) -> None:
    import pandas as pd

    if not group_cols:
        return
    g = df.groupby(group_cols, dropna=False)[metric_cols].agg(["count", "mean", "std", "min", "max"])
    g.to_csv(out)
    print(f"Wrote grouped summary -> {out}")


def write_correlation(df: Any, cols: list[str], out: Path) -> None:
    import pandas as pd

    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    c = sub.corr(numeric_only=True, min_periods=2)
    c.to_csv(out)
    print(f"Wrote correlation matrix -> {out}")


def pairwise_tests(df: Any, group_col: str, metric_cols: list[str], out: Path) -> None:
    try:
        from scipy import stats
    except ImportError:
        print("scipy not installed; skipping pairwise tests.")
        return

    import pandas as pd

    lines: list[str] = []
    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < 2:
        return
    for metric in metric_cols:
        series = pd.to_numeric(df[metric], errors="coerce")
        valid = df[[group_col]].copy()
        valid["_y"] = series
        valid = valid.dropna(subset=["_y"])
        if len(valid) < 4:
            continue
        gvals = [valid.loc[valid[group_col] == g, "_y"].values for g in groups if (valid[group_col] == g).any()]
        gvals = [v for v in gvals if len(v) > 0]
        if len(gvals) < 2:
            continue
        if len(gvals) == 2:
            t = stats.ttest_ind(gvals[0], gvals[1], equal_var=False)
            lines.append(
                f"{metric}  Welch t-test  groups={groups[:2]}  "
                f"stat={t.statistic:.4g}  p={t.pvalue:.4g}  n={[len(x) for x in gvals]}"
            )
        else:
            stacked = [valid.loc[valid[group_col] == g, "_y"].values for g in groups]
            stacked = [x for x in stacked if len(x) > 0]
            if len(stacked) >= 2:
                try:
                    f = stats.f_oneway(*stacked)
                    lines.append(
                        f"{metric}  one-way ANOVA  groups={groups}  "
                        f"F={f.statistic:.4g}  p={f.pvalue:.4g}"
                    )
                except ValueError as e:
                    lines.append(f"{metric}  ANOVA skipped: {e}")

    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if lines:
        print(f"Wrote group tests -> {out}")


def maybe_plot(df: Any, group_col: str | None, metric_cols: list[str], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    # Histograms for key graph metrics
    for m in metric_cols[:12]:
        if m not in df.columns:
            continue
        s = pd.to_numeric(df[m], errors="coerce").dropna()
        if len(s) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(s, bins=min(20, max(5, len(s) // 2)), edgecolor="black", alpha=0.75)
        ax.set_title(m)
        ax.set_xlabel(m)
        ax.set_ylabel("count")
        fig.tight_layout()
        safe = re.sub(r"[^\w\-.]+", "_", m)[:80]
        fig.savefig(out_dir / f"hist__{safe}.png", dpi=120)
        plt.close(fig)

    if group_col and group_col in df.columns:
        for m in metric_cols[:8]:
            if m not in df.columns:
                continue
            sub = df[[group_col, m]].copy()
            sub[m] = pd.to_numeric(sub[m], errors="coerce")
            sub = sub.dropna()
            if sub[group_col].nunique() < 2 or len(sub) < 4:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            for g in sub[group_col].unique():
                part = sub.loc[sub[group_col] == g, m]
                if len(part) > 0:
                    ax.scatter([g] * len(part), part, alpha=0.6, s=40)
            ax.set_xlabel(group_col)
            ax.set_ylabel(m)
            ax.set_title(f"{m} by {group_col}")
            fig.tight_layout()
            safe = re.sub(r"[^\w\-.]+", "_", m)[:60]
            fig.savefig(out_dir / f"scatter_by__{group_col}__{safe}.png", dpi=120)
            plt.close(fig)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(
        description="Combine and analyze collect_metadata.py CSV exports (per-run graph stats)."
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--folder",
        type=Path,
        metavar="DIR",
        help="Directory containing metadata CSV exports (*.csv). One row per file in the combined table.",
    )
    src.add_argument(
        "--inputs",
        nargs="+",
        metavar="CSV",
        help="One or more CSV paths (glob expanded by shell).",
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        help="With --folder: include *.csv in subfolders (default: only the given directory).",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional CSV with `file` (basename) or `path` (full path) plus config columns.",
    )
    ap.add_argument(
        "--group-by",
        default=None,
        help="Column name to group by (must come from manifest or be constant per file).",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for wide_table.csv and analysis outputs. "
        "Default: the folder passed to --folder, or ./kg_metadata_analysis for --inputs.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Write PNG histograms / scatter plots (needs matplotlib).",
    )
    args = ap.parse_args()

    if args.folder is not None:
        folder = args.folder.resolve()
        paths = discover_metadata_csvs(folder, recursive=args.recursive)
        if not paths:
            raise SystemExit(
                f"No .csv files found under {folder}"
                + (" (recursive)" if args.recursive else "")
            )
        out_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else folder
        )
    else:
        paths = [Path(p).resolve() for p in args.inputs]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"Missing files: {missing}")
        out_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else Path("kg_metadata_analysis").resolve()
        )

    rows: list[dict[str, Any]] = []
    for p in paths:
        try:
            rows.append(parse_collect_metadata_csv(p))
        except ValueError as e:
            print(f"Skipping (not a collect_metadata CSV?): {p}\n  {e}", file=sys.stderr)

    if not rows:
        raise SystemExit("No valid metadata CSVs could be parsed.")

    df = pd.DataFrame(rows)
    df = reorder_wide_columns(df)

    if args.manifest is not None:
        if not args.manifest.is_file():
            raise SystemExit(f"Manifest not found: {args.manifest}")
        man = _load_manifest(args.manifest)
        low = {c.lower() for c in man.columns}
        if "path" in low:
            df = _merge_manifest_path(rows, man)
        else:
            df = _merge_manifest_simple(rows, man)
        df = reorder_wide_columns(df)

    out_dir.mkdir(parents=True, exist_ok=True)

    wide_path = out_dir / "wide_table.csv"
    df.to_csv(wide_path, index=False)
    print(f"Wrote wide table ({len(df)} rows) -> {wide_path}")

    # Quick comparison: Neo4j DB name and node/edge counts per export file
    cmp_cols = [
        "neo4j_database",
        "basename",
        "graph__entity_nodes",
        "graph__rel_edges_directed",
        "graph__isolated_entity_count",
    ]
    have = [c for c in cmp_cols if c in df.columns]
    if have:
        cmp_df = df[have].copy()
        sort_keys = [c for c in ("neo4j_database", "basename") if c in cmp_df.columns]
        if sort_keys:
            cmp_df = cmp_df.sort_values(by=sort_keys, kind="stable")
        print("\n--- Node / edge counts by run (each row = one metadata CSV) ---")
        with pd.option_context("display.max_rows", None, "display.width", None):
            print(cmp_df.to_string(index=False))

    # Core graph metrics for stats
    preferred_metrics = [
        "graph__entity_nodes",
        "graph__rel_edges_directed",
        "graph__edge_per_node",
        "graph__isolation_rate",
        "graph__rel_type_entropy",
        "graph__entity_kind_entropy",
        "graph__rel_type_count",
        "graph__top_rel_type_share",
        "graph__unexpected_rel_total",
        "graph__unexpected_kind_total",
    ]
    num_cols = [c for c in preferred_metrics if c in df.columns]
    extra = [c for c in numeric_columns(df) if c not in num_cols and c.startswith("graph__")]
    metric_cols = num_cols + extra

    # Also include manifest numeric columns (performance scores)
    for c in df.columns:
        if c in metric_cols or c in ("source_file", "basename"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() >= 2:
            metric_cols.append(c)

    metric_cols = sorted(set(metric_cols))

    write_correlation(df, [c for c in metric_cols if c in df.columns], out_dir / "correlation_metrics.csv")

    group_col = args.group_by
    if group_col and group_col in df.columns:
        write_summary_by_group(
            df,
            [group_col],
            [c for c in metric_cols if c in df.columns],
            out_dir / "summary_by_group.csv",
        )
        pairwise_tests(
            df,
            group_col,
            [c for c in metric_cols if c in df.columns],
            out_dir / "group_tests.txt",
        )
    elif group_col:
        print(
            f"Warning: --group-by={group_col!r} not in columns {list(df.columns)}; skip grouped stats.",
            file=sys.stderr,
        )

    if args.plot:
        maybe_plot(df, group_col if group_col and group_col in df.columns else None, metric_cols, out_dir / "plots")

    # Console summary
    print("\n--- Numeric summary (graph metrics) ---")
    show = [c for c in preferred_metrics if c in df.columns]
    if show:
        print(df[show].describe().to_string())
    print("\nDone.")


if __name__ == "__main__":
    main()
