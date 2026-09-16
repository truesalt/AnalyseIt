"""
profiler.py - AnalyseIt's deterministic statistical engine.

This module is the ONLY component that ever touches raw rows. Everything
downstream (RAG queries, the LLM prompt) consumes the JSON-safe dictionary
produced here, never the DataFrame itself.

Usage:
    profiler = DataProfiler(df, name="titanic.csv")
    report = profiler.profile()                 # full dict, safe to json.dumps
    payload = profiler.to_llm_payload()         # trimmed + value-redacted dict for the LLM
    issues = profiler.top_issues(2)             # ranked issues, each with a .rag_query
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

PROFILER_VERSION = "1.0"

# --- Detection thresholds (single place to tune the engine's opinions) -------
HIGH_MISSING_PCT = 30.0        # above this, imputation is usually the wrong call
MODERATE_MISSING_PCT = 5.0
HIGH_SKEW = 1.0                # |skew| above this suggests a transform
EXTREME_SKEW = 2.0
OUTLIER_PCT_FLAG = 1.0         # % of rows outside 1.5*IQR before we complain
HIGH_CARDINALITY_RATIO = 0.5   # unique / non-null for object columns
HIGH_CARDINALITY_MIN = 50      # ...but only once there are this many levels
DOMINANT_CLASS_PCT = 95.0      # near-constant categorical
SAMPLE_SIZE = 5000             # cap for the expensive inference/normality checks


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _native(value: Any) -> Any:
    """Convert numpy/pandas scalars into plain Python so json.dumps never chokes."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if value is pd.NaT or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, str):
        return value
    return str(value)


def _round(value: Any, digits: int = 4) -> Any:
    native = _native(value)
    return round(native, digits) if isinstance(native, float) else native


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------
class DataProfiler:
    """Computes a deterministic, JSON-serializable profile of a DataFrame."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset") -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataProfiler expects a pandas DataFrame, got {type(df).__name__}")
        if df.shape[1] == 0:
            raise ValueError("DataFrame has no columns to profile.")
        self.df = df
        self.name = name
        self._report: dict[str, Any] | None = None

    # -- public API ---------------------------------------------------------
    def profile(self, force: bool = False) -> dict[str, Any]:
        """Run the full profile. Result is cached; pass force=True to recompute."""
        if self._report is not None and not force:
            return self._report

        columns = {col: self._profile_column(col) for col in self.df.columns}
        report = {
            "profiler_version": PROFILER_VERSION,
            "profiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": self._dataset_stats(columns),
            "columns": columns,
            "issues": [],
        }
        report["issues"] = self._detect_issues(report)
        self._report = report
        return report

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.profile(), indent=indent, default=str)

    def top_issues(self, k: int = 2) -> list[dict[str, Any]]:
        """Highest-severity issues first - this is what drives the RAG queries."""
        return self.profile()["issues"][:k]

    def to_llm_payload(
        self,
        max_columns: int = 40,
        max_issues: int = 8,
        include_category_values: bool = False,
    ) -> dict[str, Any]:
        """
        Compact, privacy-preserving view of the profile for the LLM prompt.

        No raw rows ever leave this function. Observed category labels are
        dropped unless `include_category_values=True` is passed explicitly,
        since labels can themselves carry PII (names, emails, IDs).
        """
        report = self.profile()
        drop_keys = {"memory_bytes"}
        if not include_category_values:
            drop_keys |= {"top_values", "mode"}

        columns: dict[str, Any] = {}
        for col, stats in list(report["columns"].items())[:max_columns]:
            columns[col] = {k: v for k, v in stats.items() if k not in drop_keys and v is not None}

        payload = {
            "dataset": report["dataset"],
            "columns": columns,
            "issues": [
                {k: v for k, v in issue.items() if k != "metrics"}
                for issue in report["issues"][:max_issues]
            ],
        }
        if len(report["columns"]) > max_columns:
            payload["note"] = (
                f"Only the first {max_columns} of {len(report['columns'])} columns are included."
            )
        return payload

    # -- dataset level ------------------------------------------------------
    def _dataset_stats(self, columns: dict[str, Any]) -> dict[str, Any]:
        n_rows, n_cols = self.df.shape
        total_cells = int(n_rows) * int(n_cols)
        missing_cells = int(self.df.isna().sum().sum())
        duplicate_rows = int(self.df.duplicated().sum()) if n_rows else 0

        type_counts: dict[str, int] = {}
        for stats in columns.values():
            key = stats["semantic_type"]
            type_counts[key] = type_counts.get(key, 0) + 1

        return {
            "name": self.name,
            "n_rows": int(n_rows),
            "n_columns": int(n_cols),
            "memory_mb": _round(self.df.memory_usage(deep=True).sum() / 1024**2, 3),
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "missing_cells_pct": _round(100 * missing_cells / total_cells, 2) if total_cells else 0.0,
            "duplicate_rows": duplicate_rows,
            "duplicate_rows_pct": _round(100 * duplicate_rows / n_rows, 2) if n_rows else 0.0,
            "column_type_counts": type_counts,
            "column_names": [str(c) for c in self.df.columns],
        }

    # -- column level -------------------------------------------------------
    def _profile_column(self, col: str) -> dict[str, Any]:
        series = self.df[col]
        n_rows = len(series)
        missing = int(series.isna().sum())
        non_null = series.dropna()
        unique = int(non_null.nunique())

        stats: dict[str, Any] = {
            "dtype": str(series.dtype),
            "semantic_type": self._semantic_type(series),
            "missing_count": missing,
            "missing_pct": _round(100 * missing / n_rows, 2) if n_rows else 0.0,
            "unique_count": unique,
            "unique_pct": _round(100 * unique / len(non_null), 2) if len(non_null) else 0.0,
            "is_constant": unique <= 1,
            "is_unique_key": unique == n_rows and missing == 0 and n_rows > 0,
            "memory_bytes": int(series.memory_usage(deep=True)),
        }

        if stats["semantic_type"] == "numeric":
            stats.update(self._numeric_stats(non_null))
        elif stats["semantic_type"] == "datetime":
            stats.update(self._datetime_stats(non_null))
        elif stats["semantic_type"] == "boolean":
            stats.update(self._boolean_stats(non_null))
        else:
            stats.update(self._categorical_stats(non_null))

        return stats

    @staticmethod
    def _semantic_type(series: pd.Series) -> str:
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_datetime64_any_dtype(dtype) or isinstance(dtype, pd.PeriodDtype):
            return "datetime"
        if pd.api.types.is_numeric_dtype(dtype):
            return "numeric"
        if isinstance(dtype, pd.CategoricalDtype):
            return "categorical"
        return "categorical"

    def _numeric_stats(self, s: pd.Series) -> dict[str, Any]:
        if s.empty:
            return {"all_missing": True}

        values = pd.to_numeric(s, errors="coerce").dropna().astype("float64")
        if values.empty:
            return {"all_missing": True}

        q1, median, q3 = (float(v) for v in values.quantile([0.25, 0.5, 0.75]))
        iqr = q3 - q1
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        mean = float(values.mean())
        n = len(values)

        # IQR (Tukey) outliers - undefined when the IQR collapses to zero.
        if iqr > 0:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_outliers = int(((values < lower) | (values > upper)).sum())
        else:
            lower = upper = None
            iqr_outliers = 0

        # Z-score outliers (|z| > 3).
        z_outliers = int((((values - mean) / std).abs() > 3).sum()) if std > 0 else 0

        stats: dict[str, Any] = {
            "mean": _round(mean),
            "std": _round(std),
            "min": _round(values.min()),
            "p01": _round(values.quantile(0.01)),
            "q1": _round(q1),
            "median": _round(median),
            "q3": _round(q3),
            "p99": _round(values.quantile(0.99)),
            "max": _round(values.max()),
            "iqr": _round(iqr),
            "range": _round(float(values.max()) - float(values.min())),
            "coefficient_of_variation": _round(std / abs(mean)) if mean != 0 else None,
            "skewness": _round(values.skew()) if n > 2 else None,
            "kurtosis": _round(values.kurt()) if n > 3 else None,
            "zero_count": int((values == 0).sum()),
            "negative_count": int((values < 0).sum()),
            "is_integer_like": bool(np.allclose(values, values.round())),
            "iqr_outlier_count": iqr_outliers,
            "iqr_outlier_pct": _round(100 * iqr_outliers / n, 2),
            "iqr_lower_bound": _round(lower),
            "iqr_upper_bound": _round(upper),
            "zscore_outlier_count": z_outliers,
            "zscore_outlier_pct": _round(100 * z_outliers / n, 2),
            "normality_p": self._normality_p(values),
        }
        return stats

    @staticmethod
    def _normality_p(values: pd.Series) -> float | None:
        """D'Agostino-Pearson p-value on a sample; None when not computable."""
        if len(values) < 20:
            return None
        try:
            from scipy import stats as scipy_stats

            sample = values.sample(SAMPLE_SIZE, random_state=0) if len(values) > SAMPLE_SIZE else values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return _round(scipy_stats.normaltest(sample.to_numpy()).pvalue, 6)
        except Exception:
            return None

    def _categorical_stats(self, s: pd.Series) -> dict[str, Any]:
        if s.empty:
            return {"all_missing": True}

        counts = s.value_counts()
        n = len(s)
        as_str = s.astype(str)
        sample = s.sample(SAMPLE_SIZE, random_state=0) if n > SAMPLE_SIZE else s

        lengths = as_str.str.len()
        stats: dict[str, Any] = {
            "mode": _native(counts.index[0]),
            "mode_freq": int(counts.iloc[0]),
            "mode_pct": _round(100 * int(counts.iloc[0]) / n, 2),
            "top_values": {str(k): int(v) for k, v in counts.head(10).items()},
            "mean_length": _round(lengths.mean(), 2),
            "max_length": int(lengths.max()),
            "empty_string_count": int((as_str.str.strip() == "").sum()),
            "mixed_python_types": len({type(v).__name__ for v in sample}) > 1,
            "looks_numeric": self._parse_ratio(sample, "numeric") > 0.9,
            "looks_datetime": self._parse_ratio(sample, "datetime") > 0.9,
        }
        return stats

    @staticmethod
    def _parse_ratio(sample: pd.Series, kind: str) -> float:
        """Fraction of values that parse cleanly as numeric / datetime."""
        if sample.empty:
            return 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "numeric":
                    parsed = pd.to_numeric(sample, errors="coerce")
                else:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            return float(parsed.notna().mean())
        except Exception:
            return 0.0

    @staticmethod
    def _datetime_stats(s: pd.Series) -> dict[str, Any]:
        if s.empty:
            return {"all_missing": True}
        values = pd.to_datetime(s, errors="coerce").dropna()
        if values.empty:
            return {"all_missing": True}
        span = values.max() - values.min()
        return {
            "min": _native(values.min()),
            "max": _native(values.max()),
            "span_days": _round(span.total_seconds() / 86400, 2),
            "is_monotonic_increasing": bool(values.is_monotonic_increasing),
        }

    @staticmethod
    def _boolean_stats(s: pd.Series) -> dict[str, Any]:
        if s.empty:
            return {"all_missing": True}
        true_count = int(s.astype("boolean").sum())
        return {
            "true_count": true_count,
            "false_count": int(len(s) - true_count),
            "true_pct": _round(100 * true_count / len(s), 2),
        }

    # -- issue detection ----------------------------------------------------
    def _detect_issues(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Rank data-quality problems by severity. Each issue carries a `rag_query`
        so rag_engine.KnowledgeBase can look up the matching best practice.
        """
        issues: list[dict[str, Any]] = []
        n_rows = report["dataset"]["n_rows"]

        def add(column, issue_type, severity, detail, rag_query, **metrics):
            issues.append(
                {
                    "column": column,
                    "issue_type": issue_type,
                    "severity_score": round(float(min(severity, 100.0)), 1),
                    "severity": self._severity_label(severity),
                    "detail": detail,
                    "rag_query": rag_query,
                    "metrics": {k: _native(v) for k, v in metrics.items()},
                }
            )

        # --- dataset-wide ---
        dup_pct = report["dataset"]["duplicate_rows_pct"]
        if report["dataset"]["duplicate_rows"] > 0:
            add(
                None,
                "duplicate_rows",
                40 + dup_pct,
                f"{report['dataset']['duplicate_rows']} duplicate rows ({dup_pct}% of the dataset).",
                "how to detect and remove duplicate rows in a dataset",
                duplicate_rows=report["dataset"]["duplicate_rows"],
                duplicate_rows_pct=dup_pct,
            )

        # Scaling is a dataset-level concern: compare the typical magnitude of
        # every numeric column and flag when they differ by orders of magnitude.
        scales = {
            col: max(abs(st.get("p01") or 0.0), abs(st.get("p99") or 0.0))
            for col, st in report["columns"].items()
            if st["semantic_type"] == "numeric" and not st.get("all_missing") and not st["is_constant"]
        }
        non_zero = {c: s for c, s in scales.items() if s > 0}
        if len(non_zero) >= 2:
            biggest, smallest = max(non_zero, key=non_zero.get), min(non_zero, key=non_zero.get)
            ratio = non_zero[biggest] / non_zero[smallest]
            if ratio > 100:
                add(
                    None,
                    "mixed_feature_scales",
                    45,
                    f"Numeric features span very different magnitudes "
                    f"('{biggest}' is ~{ratio:.0f}x the scale of '{smallest}') - "
                    f"distance- and gradient-based models need scaling.",
                    "feature scaling standardization vs min-max normalization",
                    largest_scale_column=biggest, smallest_scale_column=smallest,
                    scale_ratio=round(float(ratio), 1),
                )

        # --- per column ---
        for col, st in report["columns"].items():
            miss = st["missing_pct"]
            stype = st["semantic_type"]

            if st["is_constant"] and st["missing_pct"] < 100:
                add(col, "constant_column", 55,
                    f"'{col}' has a single distinct value and carries no signal.",
                    "zero variance constant features should be dropped before modeling",
                    unique_count=st["unique_count"])

            if miss >= 100:
                add(col, "empty_column", 95, f"'{col}' is entirely missing.",
                    "how to handle columns that are completely empty", missing_pct=miss)
            elif miss >= HIGH_MISSING_PCT:
                add(col, "high_missing", 60 + miss * 0.35,
                    f"'{col}' is {miss}% missing - imputation may fabricate signal.",
                    f"how to handle columns with a high percentage of missing {stype} values",
                    missing_pct=miss, missing_count=st["missing_count"])
            elif miss >= MODERATE_MISSING_PCT:
                add(col, "moderate_missing", 35 + miss,
                    f"'{col}' is {miss}% missing.",
                    f"best strategy for imputing missing {stype} values",
                    missing_pct=miss, missing_count=st["missing_count"])
            elif miss > 0:
                add(col, "low_missing", 15 + miss,
                    f"'{col}' has {st['missing_count']} missing values ({miss}%).",
                    f"imputing a very small percentage of missing {stype} values "
                    f"or dropping those rows",
                    missing_pct=miss, missing_count=st["missing_count"])

            if stype == "numeric" and not st.get("all_missing"):
                skew = st.get("skewness")
                if skew is not None and abs(skew) >= HIGH_SKEW:
                    sev = 45 + min(abs(skew), 5) * 8
                    label = "extremely" if abs(skew) >= EXTREME_SKEW else "moderately"
                    col_min = st.get("min")
                    add(col, "high_skew", sev,
                        f"'{col}' is {label} skewed (skewness={skew}).",
                        "log or power transform for skewed numeric features",
                        skewness=skew, min=col_min,
                        has_non_positive=(col_min is not None and col_min <= 0))

                out_pct = st.get("iqr_outlier_pct", 0.0) or 0.0
                if out_pct >= OUTLIER_PCT_FLAG:
                    add(col, "outliers", 40 + min(out_pct, 20) * 1.5,
                        f"'{col}' has {st['iqr_outlier_count']} IQR outliers ({out_pct}% of rows).",
                        "detecting outliers with the interquartile range and clipping extreme values",
                        iqr_outlier_count=st["iqr_outlier_count"], iqr_outlier_pct=out_pct,
                        lower_bound=st.get("iqr_lower_bound"), upper_bound=st.get("iqr_upper_bound"))

                # A numeric column that is unique per row is an ID, not a feature.
                if st["is_unique_key"] and st.get("is_integer_like"):
                    add(col, "identifier_like", 50,
                        f"'{col}' is a unique integer per row - it looks like an ID, not a feature.",
                        "dropping high cardinality identifier columns before modeling",
                        unique_count=st["unique_count"])

            if stype == "categorical" and not st.get("all_missing"):
                if st.get("looks_numeric"):
                    add(col, "numeric_stored_as_text", 70,
                        f"'{col}' is stored as text but parses as numeric.",
                        "converting object dtype columns to numeric dtypes",
                        dtype=st["dtype"])
                elif st.get("looks_datetime"):
                    add(col, "datetime_stored_as_text", 65,
                        f"'{col}' is stored as text but parses as a datetime.",
                        "parsing datetime columns and extracting temporal features",
                        dtype=st["dtype"])

                if st.get("mixed_python_types"):
                    add(col, "mixed_types", 60,
                        f"'{col}' mixes multiple Python types in one column.",
                        "cleaning columns with inconsistent mixed data types",
                        dtype=st["dtype"])

                if st["is_unique_key"]:
                    add(col, "identifier_like", 50,
                        f"'{col}' is unique for every row - it looks like an ID, not a feature.",
                        "dropping high cardinality identifier columns before modeling",
                        unique_count=st["unique_count"])
                elif (
                    st["unique_count"] >= HIGH_CARDINALITY_MIN
                    and st["unique_pct"] >= HIGH_CARDINALITY_RATIO * 100
                ):
                    add(col, "high_cardinality", 55,
                        f"'{col}' has {st['unique_count']} distinct levels - one-hot encoding would explode.",
                        "encoding high cardinality categorical features target or frequency encoding",
                        unique_count=st["unique_count"], unique_pct=st["unique_pct"])
                elif st["unique_count"] > 1:
                    add(col, "needs_encoding", 25,
                        f"'{col}' is categorical with {st['unique_count']} levels and needs encoding.",
                        "one hot encoding vs ordinal encoding for categorical features",
                        unique_count=st["unique_count"])

                mode_pct = st.get("mode_pct", 0.0) or 0.0
                if mode_pct >= DOMINANT_CLASS_PCT and st["unique_count"] > 1:
                    add(col, "imbalanced_category", 45,
                        f"'{col}' is dominated by one value ({mode_pct}% of rows).",
                        "handling severely imbalanced or near constant categorical features",
                        mode_pct=mode_pct)

        issues.sort(key=lambda i: (-i["severity_score"], str(i["column"])))
        for rank, issue in enumerate(issues, start=1):
            issue["rank"] = rank
        return issues

    @staticmethod
    def _severity_label(score: float) -> str:
        if score >= 70:
            return "critical"
        if score >= 45:
            return "high"
        if score >= 25:
            return "medium"
        return "low"


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def profile_csv(path: str, **read_csv_kwargs: Any) -> dict[str, Any]:
    """Read a CSV from disk and return its profile dictionary."""
    df = pd.read_csv(path, **read_csv_kwargs)
    return DataProfiler(df, name=str(path)).profile()


def _demo_frame() -> pd.DataFrame:
    """Synthetic frame containing every issue the engine knows how to detect."""
    rng = np.random.default_rng(42)
    n = 500
    demo = pd.DataFrame(
        {
            "id": range(n),
            "age": rng.normal(40, 12, n).round(),
            "income": rng.lognormal(10, 1.1, n).round(2),          # skewed + outliers
            "city": rng.choice(["Delhi", "Mumbai", "Pune", None], n, p=[0.4, 0.3, 0.2, 0.1]),
            "signup_date": pd.Series(pd.date_range("2024-01-01", periods=n, freq="D")).astype(str),
            "score_text": rng.integers(0, 100, n).astype(str),      # numeric stored as text
            "country": ["IN"] * n,                                  # constant
            "churned": rng.choice([True, False], n, p=[0.05, 0.95]),
        }
    )
    demo.loc[rng.choice(n, 200, replace=False), "age"] = np.nan     # 40% missing
    return pd.concat([demo, demo.head(5)], ignore_index=True)       # duplicates


def _cli() -> int:
    """
    python profiler.py                 -> profile the built-in demo dataset
    python profiler.py data.csv        -> profile your own CSV
    python profiler.py data.csv --json -> dump the full profile as JSON
    """
    import argparse

    parser = argparse.ArgumentParser(description="AnalyseIt statistical profiler")
    parser.add_argument("csv", nargs="?", help="path to a CSV file (omit to use the demo dataset)")
    parser.add_argument("--json", action="store_true", help="print the full profile as JSON")
    parser.add_argument("--top", type=int, default=10, help="how many issues to list (default 10)")
    args = parser.parse_args()

    if args.csv:
        try:
            df = pd.read_csv(args.csv)
        except FileNotFoundError:
            print(f"ERROR: no such file: {args.csv}")
            return 1
        except Exception as exc:
            print(f"ERROR: could not read {args.csv}: {exc}")
            return 1
        name = args.csv
    else:
        df, name = _demo_frame(), "demo.csv (built-in synthetic dataset)"

    profiler = DataProfiler(df, name=name)
    report = profiler.profile()

    if args.json:
        print(profiler.to_json())
        return 0

    ds = report["dataset"]
    print(f"\n{'=' * 78}\n  {ds['name']}\n{'=' * 78}")
    print(f"  {ds['n_rows']:,} rows x {ds['n_columns']} columns   |   {ds['memory_mb']} MB in memory")
    print(f"  {ds['missing_cells']:,} missing cells ({ds['missing_cells_pct']}%)   |   "
          f"{ds['duplicate_rows']:,} duplicate rows ({ds['duplicate_rows_pct']}%)")
    print(f"  column types: {ds['column_type_counts']}")

    issues = report["issues"]
    print(f"\n  {len(issues)} ISSUES DETECTED (top {min(args.top, len(issues))} shown)\n  {'-' * 74}")
    for issue in issues[: args.top]:
        col = issue["column"] or "<dataset>"
        print(f"  {issue['rank']:>2}. [{issue['severity']:>8}] {col:<16} {issue['issue_type']}")
        print(f"      {issue['detail']}")

    payload_chars = len(json.dumps(profiler.to_llm_payload()))
    print(f"\n  LLM payload: {payload_chars:,} chars (~{payload_chars // 4:,} tokens), 0 raw rows.")
    print(f"  Run with --json for the full profile.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
