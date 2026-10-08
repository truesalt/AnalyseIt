"""
eval_retrieval.py - measure AnalyseIt's retrieval against every query the profiler emits.

    python eval_retrieval.py                   # default backend (ANALYSEIT_EMBED_BACKEND or auto)
    python eval_retrieval.py --backend onnx
    python eval_retrieval.py --compare         # also compare sentence-transformers with ONNX
    python eval_retrieval.py --memory          # also measure peak RSS of each backend

Checks (the script exits non-zero if any fails):
  1. Chunk integrity - every chunk starts with its section heading and ends on a
     sentence boundary.
  2. Top-1 accuracy - each rag_query the profiler emits for a CSV upload must
     retrieve, first, the section written for that issue. The queries are not
     hard-coded: they are collected by writing a synthetic frame that triggers
     every issue type to CSV and profiling what read_csv gives back, so this
     cannot drift from profiler.py and proves every type fires for an upload.
  3. Relevance cutoff - every expected passage clears MIN_SIMILARITY, and
     off-topic queries retrieve nothing.
  4. (--compare) Backend agreement - identical rankings, and the largest
     per-component difference between the two backends' vectors.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from profiler import DataProfiler, read_csv_bytes
from rag_engine import DEFAULT_DOCS_FILE, MIN_SIMILARITY, KnowledgeBase, chunk_document

# The knowledge-base section each issue type is meant to retrieve.
EXPECTED_SECTION = {
    "duplicate_rows": "Detecting and removing duplicate rows",
    "mixed_feature_scales": "Feature scaling: standardization versus normalization",
    "constant_column": "Zero variance and constant features",
    "empty_column": "Handling columns that are completely empty",
    "low_missing": "Imputing a small number of missing values",
    "high_skew": "Transforming skewed numeric features",
    "outliers": "Detecting and treating outliers",
    "identifier_like": "Dropping identifier columns before modeling",
    "numeric_stored_as_text": "Converting text columns to numeric dtypes",
    "datetime_stored_as_text": "Parsing datetime columns and extracting features",
    "mixed_types": "Cleaning columns with mixed data types",
    "high_cardinality": "Encoding high cardinality categorical features",
    "needs_encoding": "Encoding low cardinality categorical features",
    "imbalanced_category": "Handling imbalanced and near constant categorical features",
    "sentinel_values": "Placeholder codes and infinite values in numeric columns",
    "infinite_values": "Placeholder codes and infinite values in numeric columns",
}

OFF_TOPIC_QUERIES = [
    "best pizza toppings in naples",
    "how do I configure a kubernetes ingress controller",
    "weather forecast for tomorrow",
    "history of the roman empire",
]


def expected_section(issue_type: str, query: str) -> str:
    if issue_type in ("high_missing", "moderate_missing"):
        kind = "numeric" if "numeric" in query else "categorical"
        return f"Handling missing values in {kind} columns"
    return EXPECTED_SECTION[issue_type]


def _issue_zoo(n: int = 600) -> pd.DataFrame:
    """A frame that triggers every issue type, with missing values in both a
    numeric and a categorical column at each missingness tier."""
    rng = np.random.default_rng(0)
    visits = rng.integers(0, 40, n).astype(float)
    visits[rng.choice(n, n // 20, replace=False)] = -999                   # placeholder code
    ratio = rng.uniform(0.5, 2.0, n)
    ratio[rng.choice(n, n // 50, replace=False)] = np.inf                  # infinite values
    df = pd.DataFrame({
        "id": np.arange(n),
        "income": rng.lognormal(10, 1.2, n).round(2),                      # skew, outliers, scale
        "age": rng.normal(40, 12, n).round(),
        "score": rng.uniform(0, 1, n),
        "visits": visits,
        "ratio": ratio,
        "segment": rng.choice(["a", "b", "c"], n),
        "city": rng.choice(["Delhi", "Mumbai", "Pune", "Chennai"], n),
        "plan": rng.choice(["basic", "pro"], n, p=[0.97, 0.03]),           # imbalanced
        "country": ["IN"] * n,                                             # constant
        "notes": [None] * n,                                               # empty
        "signup": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "amount": [f"${v:,}" for v in rng.integers(0, 5000, n)],           # numeric as text
        "sku": [f"SKU-{i}" for i in rng.integers(0, 80, n)],               # high cardinality
        "mixed": [1, "a", 2.5] * (n // 3),                                 # mixed types
    })
    for column, frac in [("income", 0.35), ("segment", 0.40),             # high missing
                         ("age", 0.10), ("city", 0.10),                    # moderate
                         ("score", 0.02), ("plan", 0.02)]:                 # low
        df.loc[rng.choice(n, int(n * frac), replace=False), column] = None
    return pd.concat([df, df.head(3)], ignore_index=True)                 # duplicates


def profiler_queries() -> list[tuple[str, str]]:
    """Every distinct (issue_type, rag_query) the profiler emits for the zoo
    frame after a round trip through CSV - i.e. for an actual upload."""
    df, _, _ = read_csv_bytes(_issue_zoo().to_csv(index=False).encode("utf-8"))
    issues = DataProfiler(df).profile()["issues"]
    return sorted({(i["issue_type"], i["rag_query"]) for i in issues})


def check_chunks() -> dict[str, Any]:
    chunks = chunk_document(DEFAULT_DOCS_FILE.read_text(encoding="utf-8"))
    return {
        "chunks": len(chunks),
        "sections": len({c["section"] for c in chunks}),
        "missing_heading": sum(not c["text"].startswith(c["section"] + "\n\n") for c in chunks),
        "mid_sentence_cuts": sum(not c["text"].rstrip().endswith((".", "!", "?", '"', ")")) for c in chunks),
    }


def run_eval(backend: str | None = None, top_k: int = 3) -> dict[str, Any]:
    """Run checks 1-3 against an in-memory index. Returns a JSON-safe report."""
    kb = KnowledgeBase(backend=backend, in_memory=True, collection_name="eval_retrieval")
    kb.build_index()

    rows = []
    for issue_type, query in profiler_queries():
        want = expected_section(issue_type, query)
        hits = kb.retrieve_context(query, top_k=top_k, min_similarity=0.0)
        rows.append({
            "issue_type": issue_type,
            "query": query,
            "expected": want,
            "top1": hits[0]["section"],
            "top1_similarity": hits[0]["similarity"],
            "correct": hits[0]["section"] == want,
            "ranking": [h["section"] for h in hits],
        })

    off_topic = {q: kb.retrieve_context(q, top_k=top_k) for q in OFF_TOPIC_QUERIES}
    off_topic_max = max(
        kb.retrieve_context(q, top_k=1, min_similarity=0.0)[0]["similarity"] for q in OFF_TOPIC_QUERIES
    )
    chunks = check_chunks()
    report = {
        "backend": kb.backend,
        "chunks": chunks,
        "queries": rows,
        "issue_types_covered": len({r["issue_type"] for r in rows}),
        "top1_correct": sum(r["correct"] for r in rows),
        "min_expected_similarity": min(r["top1_similarity"] for r in rows),
        "min_similarity_cutoff": MIN_SIMILARITY,
        "off_topic_max_similarity": off_topic_max,
        "off_topic_hits": sum(len(h) for h in off_topic.values()),
    }
    report["passed"] = (
        chunks["missing_heading"] == 0
        and chunks["mid_sentence_cuts"] == 0
        and report["issue_types_covered"] == len(EXPECTED_SECTION) + 2
        and report["top1_correct"] == len(rows)
        and report["min_expected_similarity"] >= MIN_SIMILARITY
        and report["off_topic_hits"] == 0
    )
    return report


def compare_backends(top_k: int = 3) -> dict[str, Any]:
    """Check 4: embed every chunk and query with both backends."""
    from rag_engine import _Embedder

    texts = [c["text"] for c in chunk_document(DEFAULT_DOCS_FILE.read_text(encoding="utf-8"))]
    texts += [q for _, q in profiler_queries()]
    st = np.array(_Embedder("sentence-transformers").embed(texts))
    onnx = np.array(_Embedder("onnx").embed(texts))
    rankings = {b: [r["ranking"] for r in run_eval(b, top_k)["queries"]] for b in ("sentence-transformers", "onnx")}
    return {
        "max_abs_component_diff": float(np.abs(st - onnx).max()),
        "min_cosine_between_backends": float((st * onnx).sum(axis=1).min()),
        "byte_identical": bool((st == onnx).all()),
        "rankings_identical": rankings["sentence-transformers"] == rankings["onnx"],
    }


_MEMORY_PROBE = """
import resource, sys
from rag_engine import KnowledgeBase
kb = KnowledgeBase(backend=sys.argv[1], in_memory=True, collection_name="mem_probe")
kb.build_index()
kb.retrieve_context("how to handle missing numeric values")
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak if sys.platform == "darwin" else peak * 1024)
"""


def measure_memory(backend: str) -> float:
    """Peak RSS in MB of a fresh process that builds the index and runs one query."""
    out = subprocess.run([sys.executable, "-c", _MEMORY_PROBE, backend], cwd=Path(__file__).parent,
                         capture_output=True, text=True, check=True)
    return int(out.stdout.strip().splitlines()[-1]) / 1024**2


def main() -> int:
    parser = argparse.ArgumentParser(description="AnalyseIt retrieval evaluation")
    parser.add_argument("--backend", choices=["sentence-transformers", "onnx"])
    parser.add_argument("--compare", action="store_true", help="compare both embedding backends")
    parser.add_argument("--memory", action="store_true", help="measure peak RSS per backend")
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = parser.parse_args()
    if (args.compare or args.memory) and importlib.util.find_spec("sentence_transformers") is None:
        print("--compare and --memory need sentence-transformers (pip install -r requirements-dev.txt).")
        return 2

    report = run_eval(args.backend)
    if args.compare:
        report["backend_comparison"] = compare_backends()
    if args.memory:
        report["peak_rss_mb"] = {b: round(measure_memory(b), 1) for b in ("sentence-transformers", "onnx")}

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        c = report["chunks"]
        print(f"backend: {report['backend']}")
        print(f"chunks: {c['chunks']} in {c['sections']} sections | missing headings: "
              f"{c['missing_heading']} | mid-sentence cuts: {c['mid_sentence_cuts']}")
        print(f"top-1 accuracy: {report['top1_correct']}/{len(report['queries'])} queries "
              f"({report['issue_types_covered']} issue types)")
        for r in report["queries"]:
            mark = "ok  " if r["correct"] else "MISS"
            print(f"  {mark} [{r['top1_similarity']:.3f}] {r['query'][:62]:62s} -> {r['top1']}")
        print(f"relevance cutoff {report['min_similarity_cutoff']}: lowest expected passage "
              f"{report['min_expected_similarity']:.3f}, highest off-topic "
              f"{report['off_topic_max_similarity']:.3f}, off-topic hits {report['off_topic_hits']}")
        if "backend_comparison" in report:
            b = report["backend_comparison"]
            print(f"backends: max component diff {b['max_abs_component_diff']:.1e}, min cosine "
                  f"{b['min_cosine_between_backends']:.7f}, byte-identical {b['byte_identical']}, "
                  f"rankings identical {b['rankings_identical']}")
        if "peak_rss_mb" in report:
            print("peak RSS (MB): " + ", ".join(f"{k} {v}" for k, v in report["peak_rss_mb"].items()))
        print("PASSED" if report["passed"] else "FAILED")

    ok = report["passed"] and report.get("backend_comparison", {}).get("rankings_identical", True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
