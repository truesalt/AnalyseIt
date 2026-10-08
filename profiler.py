"""
profiler.py - AnalyseIt's deterministic statistical engine.

This module is the ONLY component that ever touches raw rows. Everything
downstream (RAG queries, the LLM prompt) consumes the JSON-safe dictionary
produced here, never the DataFrame itself.

Usage:
    profiler = DataProfiler(df, name="titanic.csv", target="survived")
    report = profiler.profile()                 # full dict, safe to json.dumps
    payload = profiler.to_llm_payload()         # trimmed + value-redacted dict for the LLM
    issues = profiler.top_issues(2)             # ranked issues, each with a .rag_query
"""

from __future__ import annotations

import io
import json
import math
import warnings
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

PROFILER_VERSION = "1.1"

# --- Detection thresholds (single place to tune the engine's opinions) -------
HIGH_MISSING_PCT = 30.0        # above this, imputation is usually the wrong call
MODERATE_MISSING_PCT = 5.0
HIGH_SKEW = 1.0                # |skew| above this suggests a transform
EXTREME_SKEW = 2.0
OUTLIER_PCT_FLAG = 1.0         # % of rows outside 1.5*IQR before we complain
HIGH_CARDINALITY_MIN = 50      # levels beyond which one-hot encoding stops being viable
KEY_MIN_UNIQUE_RATIO = 0.9     # an ID column survives a few duplicate rows, not many
DOMINANT_CLASS_PCT = 95.0      # near-constant categorical
MIXED_NUMERIC_RANGE = (0.1, 0.9)   # share of numeric values that makes a text column "mixed"
MIN_COLUMNS_FOR_DUPLICATES = 3     # in 1-2 columns, repeated rows are just repeated values
SAMPLE_SIZE = 5000             # cap for the expensive inference/normality checks

# Numbers that legacy systems and surveys write instead of leaving a cell empty.
# One is treated as a placeholder only when it covers SENTINEL_MIN_PCT of the
# values and sits outside the range of all the others; -1 and -9 are plausible
# real values, so they also need the others to have at least 3 distinct values.
SENTINEL_CODES = (-1, -9, -99, -999, -9999, 999, 9999, 99999, 999999)
SENTINEL_MIN_PCT = 1.0

# A real date carries a 4-digit year or a d/m/yy shape. dateutil alone also
# reads month names ("May"), times ("10:30") and fractions ("3/4") as dates.
_DATE_SHAPE = r"\d{4}|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2}\b"

# Strings that mean "missing" in hand-maintained files. pandas already reads
# "", "NA", "N/A", "null", "NaN" and friends as NaN; these survive as text.
MISSING_PLACEHOLDERS = frozenset(
    {"", "-", "--", "?", "na", "n/a", "nan", "null", "none", "missing", "unknown"}
)
# Decoration that stops a number parsing: a comma followed by exactly three
# digits (thousands separator), currency symbols, percent signs, whitespace.
# Deliberately narrower than "[^0-9.-]", which would read "SKU-0001" as -1.
_THOUSANDS_SEPARATOR = r"(?<=\d),(?=\d{3}(?!\d))"
_NUMBER_DECORATION = r"[\s$€£¥₹%]"
_ACCOUNTING_NEGATIVE = r"^\((.+)\)$"   # "(1,200)" -> "-1,200"
# "523,45" or "1.234,56": a decimal comma, the norm wherever ";" separates fields.
_DECIMAL_COMMA = r"-?\d{1,3}(?:\.\d{3})+,\d+|-?\d+,\d+"

# --- What may reach the LLM ---------------------------------------------------
# Observed category labels are values, not statistics, and can carry PII.
CATEGORY_VALUE_KEYS = frozenset({"top_values", "mode"})
# Kept in the local profile, but no issue uses them, so they are not worth tokens.
LOCAL_ONLY_KEYS = frozenset({
    "memory_bytes", "normality_p", "kurtosis", "coefficient_of_variation",
    "zscore_outlier_count", "zscore_outlier_pct", "empty_string_count", "distinct_real_values",
})
MAX_LLM_COLUMNS = 40           # column statistics per request; bounds prompt size
MAX_LLM_COLUMN_NAMES = 200

# UTF-8 (with or without a BOM), then what Excel on Windows writes, then
# latin-1, which decodes any byte sequence.
CSV_ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")
CSV_DELIMITERS = (b",", b";", b"\t", b"|")   # ties go to the first: a comma


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _native(value: Any) -> Any:
    """Convert numpy/pandas scalars into plain Python so json.dumps never chokes."""
    # Before the datetime check: pd.NaT is a datetime subclass and would come
    # out as the string "NaT"; pd.NA would fall through to "<NA>".
    if value is None or value is pd.NaT or value is pd.NA:
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


def _code_label(code: float) -> str:
    """-999.0 -> "-999", for JSON keys and issue text."""
    return str(int(code)) if float(code).is_integer() else str(code)


def display_name(column: Any, limit: int = 60) -> str:
    """
    A column name made safe to embed in prose that reaches the LLM: one line,
    no quote characters, bounded length. Column names come from the uploaded
    file, so a newline in one could otherwise start a fake instruction. The
    exact name still travels, JSON-escaped, in the statistics.
    """
    text = " ".join(str(column).split()).translate(str.maketrans("", "", "'\"`"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------
class DataProfiler:
    """Computes a deterministic, JSON-serializable profile of a DataFrame."""

    def __init__(
        self, df: pd.DataFrame, name: str = "dataset", target: str | None = None
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataProfiler expects a pandas DataFrame, got {type(df).__name__}")
        if df.shape[1] == 0:
            raise ValueError("DataFrame has no columns to profile.")
        if target is not None and target not in df.columns:
            raise ValueError(f"Target column {target!r} is not in the DataFrame.")
        self.df = df
        self.name = name
        self.target = target
        self._duplicate_rows = 0
        self._report: dict[str, Any] | None = None

    # -- public API ---------------------------------------------------------
    def profile(self, force: bool = False) -> dict[str, Any]:
        """Run the full profile. Result is cached; pass force=True to recompute."""
        if self._report is not None and not force:
            return self._report

        # Needed before the columns: duplicated rows repeat their ID, so a key
        # column is only recognisable against the count of distinct rows.
        self._duplicate_rows = int(self.df.duplicated().sum()) if len(self.df) else 0
        columns = {col: self._profile_column(col) for col in self.df.columns}
        report = {
            "profiler_version": PROFILER_VERSION,
            "profiled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": self._dataset_stats(columns),
            "columns": columns,
            "issues": [],
        }
        report["issues"] = self.detect_issues(report)
        self._report = report
        return report

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.profile(), indent=indent, default=str)

    def top_issues(self, k: int = 2) -> list[dict[str, Any]]:
        """Highest-severity issues first - this is what drives the RAG queries."""
        return self.profile()["issues"][:k]

    def to_llm_payload(
        self,
        max_columns: int = MAX_LLM_COLUMNS,
        max_issues: int = 8,
        include_category_values: bool = False,
    ) -> dict[str, Any]:
        """
        Compact, privacy-preserving view of the profile for the LLM prompt:
        llm_view() of the whole profile plus the top issues. See llm_view()
        for exactly what is removed.
        """
        report = self.profile()
        payload = llm_view(
            report, max_columns=max_columns, include_category_values=include_category_values
        )
        payload["issues"] = [
            {k: v for k, v in issue.items() if k != "metrics"}
            for issue in report["issues"][:max_issues]
        ]
        return payload

    # -- dataset level ------------------------------------------------------
    def _dataset_stats(self, columns: dict[str, Any]) -> dict[str, Any]:
        n_rows, n_cols = self.df.shape
        total_cells = int(n_rows) * int(n_cols)
        missing_cells = int(self.df.isna().sum().sum())
        duplicate_rows = self._duplicate_rows

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
            "target": self.target,
        }

    # -- column level -------------------------------------------------------
    def _profile_column(self, col: str) -> dict[str, Any]:
        series = self.df[col]
        n_rows = len(series)
        missing = int(series.isna().sum())
        non_null = series.dropna()
        unique = int(non_null.nunique())
        distinct_rows = n_rows - self._duplicate_rows

        stats: dict[str, Any] = {
            "dtype": str(series.dtype),
            "semantic_type": self._semantic_type(series),
            "missing_count": missing,
            "missing_pct": _round(100 * missing / n_rows, 2) if n_rows else 0.0,
            "unique_count": unique,
            "unique_pct": _round(100 * unique / len(non_null), 2) if len(non_null) else 0.0,
            # Exactly one value. A column with no values at all (all missing, or
            # a header-only file) is not "constant".
            "is_constant": unique == 1,
            # Unique per *distinct* row, because accidental duplicate rows repeat
            # the key. Only while duplicates are rare: when most rows repeat, any
            # column is "unique" among the few distinct rows left.
            "is_unique_key": (
                missing == 0
                and unique > 1
                and unique == distinct_rows
                and unique >= KEY_MIN_UNIQUE_RATIO * n_rows
            ),
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
        # Set aside what is not a measurement before describing the distribution:
        # a single inf turns mean, std, max and skew into None, and -999-style
        # placeholder codes manufacture skew and outliers. Both become issues.
        infinite = np.isinf(values)
        values = values[~infinite]
        sentinels = self._sentinel_codes(values)
        values = values[~values.isin(list(sentinels))]
        flagged: dict[str, Any] = {}
        if infinite.any():
            flagged["infinite_count"] = int(infinite.sum())
        if sentinels:
            flagged["sentinel_values"] = {_code_label(c): k for c, k in sentinels.items()}
            flagged["sentinel_pct"] = _round(100 * sum(sentinels.values()) / len(s), 2)
        if values.empty:
            return {"all_missing": True, **flagged}

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
            # unique_count still counts inf and placeholder codes; this does not.
            "distinct_real_values": int(values.nunique()),
            **flagged,
        }
        return stats

    @staticmethod
    def _sentinel_codes(values: pd.Series) -> dict[float, int]:
        """Placeholder codes among finite values: frequent, and apart from the real values."""
        if values.empty:
            return {}
        counts = values[values.isin(SENTINEL_CODES)].value_counts()
        frequent = {float(c): int(k) for c, k in counts.items() if 100 * k / len(values) >= SENTINEL_MIN_PCT}
        others = values[~values.isin(list(frequent))]
        if not frequent or others.empty:
            return {}
        lo, hi, levels = float(others.min()), float(others.max()), others.nunique()
        return {
            code: k for code, k in frequent.items()
            if (code < lo or code > hi) and (abs(code) >= 99 or levels >= 3)
        }

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
        stripped = as_str.str.strip()
        sample = s.sample(SAMPLE_SIZE, random_state=0) if n > SAMPLE_SIZE else s

        # "$1,200" and "N/A"-style text keep a numeric column stored as text,
        # so numbers are judged after stripping decoration, ignoring placeholders.
        raw_numeric = self._parse_ratio(sample, "numeric")
        clean_numeric = self._parse_ratio(sample, "numeric", strip_decoration=True)

        lengths = as_str.str.len()
        stats: dict[str, Any] = {
            "mode": _native(counts.index[0]),
            "mode_freq": int(counts.iloc[0]),
            "mode_pct": _round(100 * int(counts.iloc[0]) / n, 2),
            "top_values": {str(k): int(v) for k, v in counts.head(10).items()},
            "mean_length": _round(lengths.mean(), 2),
            "max_length": int(lengths.max()),
            "empty_string_count": int((stripped == "").sum()),
            "missing_placeholder_count": int(stripped.str.lower().isin(MISSING_PLACEHOLDERS).sum()),
            # A CSV reader hands back every non-numeric column as strings, so a
            # column of "1", "hello", "3" only shows as mixed in this share.
            "numeric_fraction": _round(clean_numeric, 3),
            "mixed_python_types": len({type(v).__name__ for v in sample}) > 1,
            "looks_numeric": clean_numeric > 0.9,
            # True when some values only parse once symbols/separators are removed,
            # i.e. a plain pd.to_numeric(errors="coerce") would destroy them.
            "numeric_needs_cleaning": clean_numeric > 0.9 and raw_numeric < clean_numeric,
            "looks_datetime": self._parse_ratio(sample, "datetime") > 0.9,
        }
        return stats

    @staticmethod
    def _parse_ratio(sample: pd.Series, kind: str, strip_decoration: bool = False) -> float:
        """Fraction of non-placeholder values that parse as numeric / datetime."""
        text = sample.astype(str).str.strip()
        text = text[~text.str.lower().isin(MISSING_PLACEHOLDERS)]
        if text.empty:
            return 0.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "numeric":
                    if strip_decoration:
                        text = (
                            text.str.replace(_ACCOUNTING_NEGATIVE, r"-\1", regex=True)
                            .str.replace(_THOUSANDS_SEPARATOR, "", regex=True)
                            .str.replace(_NUMBER_DECORATION, "", regex=True)
                        )
                        european = text.str.fullmatch(_DECIMAL_COMMA)
                        text = text.where(
                            ~european,
                            text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                        )
                    parsed = pd.to_numeric(text, errors="coerce")
                else:
                    parsed = pd.to_datetime(text, errors="coerce", format="mixed")
                    parsed = parsed.where(text.str.contains(_DATE_SHAPE, regex=True))
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
    @staticmethod
    def detect_issues(report: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Rank data-quality problems by severity. Each issue carries a `rag_query`
        so rag_engine.KnowledgeBase can look up the matching best practice.

        Works from the report dict alone, so issues can be re-derived for a new
        target column without re-profiling. The target is the label, not a
        feature, so it is left out of every issue.
        """
        issues: list[dict[str, Any]] = []
        target = report["dataset"].get("target")
        features = {col: st for col, st in report["columns"].items() if col != target}

        def add(column, issue_type, severity, detail, rag_query, **metrics):
            issues.append(
                {
                    "column": column,
                    "issue_type": issue_type,
                    "severity_score": round(float(min(severity, 100.0)), 1),
                    "severity": DataProfiler._severity_label(severity),
                    "detail": detail,
                    "rag_query": rag_query,
                    "metrics": {k: _native(v) for k, v in metrics.items()},
                }
            )

        # --- dataset-wide ---
        # With one or two columns, repeated rows are just repeated values.
        dup_pct = report["dataset"]["duplicate_rows_pct"]
        if (
            report["dataset"]["duplicate_rows"] > 0
            and report["dataset"]["n_columns"] >= MIN_COLUMNS_FOR_DUPLICATES
        ):
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
        # Binary 0/1 indicators (and constants) have no meaningful scale, and an
        # ID column is flagged for dropping, not scaling.
        scales = {
            col: max(abs(st.get("p01") or 0.0), abs(st.get("p99") or 0.0))
            for col, st in features.items()
            if st["semantic_type"] == "numeric"
            and st.get("distinct_real_values", 0) > 2
            and not (st["is_unique_key"] and st.get("is_integer_like"))
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
                    f"('{display_name(biggest)}' is ~{ratio:.0f}x the scale of "
                    f"'{display_name(smallest)}') - distance- and gradient-based models need scaling.",
                    "feature scaling standardization vs min-max normalization",
                    largest_scale_column=biggest, smallest_scale_column=smallest,
                    scale_ratio=round(float(ratio), 1),
                )

        # --- per column ---
        for col, st in features.items():
            miss = st["missing_pct"]
            stype = st["semantic_type"]
            name = display_name(col)  # issue text reaches the LLM prompt

            if st["is_constant"]:
                add(col, "constant_column", 55,
                    f"'{name}' has a single distinct value and carries no signal.",
                    "zero variance constant features should be dropped before modeling",
                    unique_count=st["unique_count"])

            if miss >= 100:
                add(col, "empty_column", 95, f"'{name}' is entirely missing.",
                    "how to handle columns that are completely empty", missing_pct=miss)
            elif miss >= HIGH_MISSING_PCT:
                add(col, "high_missing", 60 + miss * 0.35,
                    f"'{name}' is {miss}% missing - imputation may fabricate signal.",
                    f"how to handle columns with a high percentage of missing {stype} values",
                    missing_pct=miss, missing_count=st["missing_count"])
            elif miss >= MODERATE_MISSING_PCT:
                add(col, "moderate_missing", 35 + miss,
                    f"'{name}' is {miss}% missing.",
                    f"best strategy for imputing missing {stype} values",
                    missing_pct=miss, missing_count=st["missing_count"])
            elif miss > 0:
                add(col, "low_missing", 15 + miss,
                    f"'{name}' has {st['missing_count']} missing values ({miss}%).",
                    f"imputing a very small percentage of missing {stype} values "
                    f"or dropping those rows",
                    missing_pct=miss, missing_count=st["missing_count"])

            if stype == "numeric" and st.get("infinite_count"):
                add(col, "infinite_values", 75,
                    f"'{name}' contains {st['infinite_count']} infinite values (inf or -inf); "
                    f"most models reject them outright.",
                    "replacing infinite values in a numeric column with missing values",
                    infinite_count=st["infinite_count"])
            if stype == "numeric" and st.get("sentinel_values"):
                codes = ", ".join(st["sentinel_values"])
                add(col, "sentinel_values", 72,
                    f"'{name}' holds placeholder code(s) {codes} ({st['sentinel_pct']}% of values) - "
                    f"missing values in disguise, so treating them as outliers would be wrong. "
                    f"The other statistics for this column exclude them.",
                    "numeric placeholder codes like -999 or 99999 that stand for missing values",
                    sentinel_values=st["sentinel_values"], sentinel_pct=st["sentinel_pct"])

            # Binary 0/1 columns are already-encoded indicators: their "skew" and
            # "outliers" only restate the class balance.
            if stype == "numeric" and st.get("distinct_real_values", 0) > 2:
                skew = st.get("skewness")
                if skew is not None and abs(skew) >= HIGH_SKEW:
                    sev = 45 + min(abs(skew), 5) * 8
                    label = "extremely" if abs(skew) >= EXTREME_SKEW else "moderately"
                    col_min = st.get("min")
                    add(col, "high_skew", sev,
                        f"'{name}' is {label} skewed (skewness={skew}).",
                        "log or power transform for skewed numeric features",
                        skewness=skew, min=col_min,
                        has_non_positive=(col_min is not None and col_min <= 0))

                out_pct = st.get("iqr_outlier_pct", 0.0) or 0.0
                if out_pct >= OUTLIER_PCT_FLAG:
                    add(col, "outliers", 40 + min(out_pct, 20) * 1.5,
                        f"'{name}' has {st['iqr_outlier_count']} IQR outliers ({out_pct}% of rows).",
                        "detecting outliers with the interquartile range and clipping extreme values",
                        iqr_outlier_count=st["iqr_outlier_count"], iqr_outlier_pct=out_pct,
                        lower_bound=st.get("iqr_lower_bound"), upper_bound=st.get("iqr_upper_bound"))

                # A numeric column that is unique per row is an ID, not a feature.
                if st["is_unique_key"] and st.get("is_integer_like"):
                    add(col, "identifier_like", 50,
                        f"'{name}' is a unique integer per row - it looks like an ID, not a feature.",
                        "dropping high cardinality identifier columns before modeling",
                        unique_count=st["unique_count"])

            if stype == "categorical" and not st.get("all_missing"):
                if st.get("looks_numeric"):
                    detail = f"'{name}' is stored as text but parses as numeric"
                    if st.get("numeric_needs_cleaning"):
                        detail += (" once currency symbols, thousands separators, % signs or "
                                   "decimal commas are handled (plain coercion would turn those "
                                   "values into NaN)")
                    detail += "."
                    if st.get("missing_placeholder_count"):
                        detail += (f" {st['missing_placeholder_count']} values are text placeholders "
                                   f"for missing (such as '-' or 'unknown').")
                    add(col, "numeric_stored_as_text", 70, detail,
                        "converting object dtype columns to numeric dtypes",
                        dtype=st["dtype"], needs_cleaning=st.get("numeric_needs_cleaning"),
                        missing_placeholder_count=st.get("missing_placeholder_count"))
                elif st.get("looks_datetime"):
                    add(col, "datetime_stored_as_text", 65,
                        f"'{name}' is stored as text but parses as a datetime.",
                        "parsing datetime columns and extracting temporal features",
                        dtype=st["dtype"])

                low, high = MIXED_NUMERIC_RANGE
                numeric_share = st.get("numeric_fraction") or 0.0
                if st.get("mixed_python_types") or low < numeric_share <= high:
                    detail = (
                        f"'{name}' mixes multiple Python types in one column."
                        if st.get("mixed_python_types")
                        else f"'{name}' mixes numbers and text: {numeric_share:.0%} of its values are numeric."
                    )
                    add(col, "mixed_types", 60, detail,
                        "cleaning columns with inconsistent mixed data types",
                        dtype=st["dtype"], numeric_fraction=numeric_share)

                # Text that is really numbers or dates gets converted, not encoded;
                # ID/encoding advice for it would contradict the conversion.
                if not (st.get("looks_numeric") or st.get("looks_datetime")):
                    if st["is_unique_key"]:
                        add(col, "identifier_like", 50,
                            f"'{name}' is unique for every row - it looks like an ID, not a feature.",
                            "dropping high cardinality identifier columns before modeling",
                            unique_count=st["unique_count"])
                    elif st["unique_count"] >= HIGH_CARDINALITY_MIN:
                        add(col, "high_cardinality", 55,
                            f"'{name}' has {st['unique_count']} distinct levels - one-hot encoding would explode.",
                            "encoding high cardinality categorical features target or frequency encoding",
                            unique_count=st["unique_count"], unique_pct=st["unique_pct"])
                    elif st["unique_count"] > 1:
                        add(col, "needs_encoding", 25,
                            f"'{name}' is categorical with {st['unique_count']} levels and needs encoding.",
                            "one hot encoding vs ordinal encoding for categorical features",
                            unique_count=st["unique_count"])

                mode_pct = st.get("mode_pct", 0.0) or 0.0
                if mode_pct >= DOMINANT_CLASS_PCT and st["unique_count"] > 1:
                    add(col, "imbalanced_category", 45,
                        f"'{name}' is dominated by one value ({mode_pct}% of rows).",
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
# Working with a finished report
# ---------------------------------------------------------------------------
def with_target(report: dict[str, Any], target: str | None) -> dict[str, Any]:
    """
    Copy of `report` with `target` marked as the label column and the issues
    re-derived without it. No statistics are recomputed, so a UI can change
    the target without re-profiling.
    """
    if target is not None and target not in report["columns"]:
        raise ValueError(f"Target column {target!r} is not in the profile.")
    updated = {**report, "dataset": {**report["dataset"], "target": target}}
    updated["issues"] = DataProfiler.detect_issues(updated)
    return updated


def llm_view(
    report: dict[str, Any],
    columns: list[str] | None = None,
    max_columns: int = MAX_LLM_COLUMNS,
    include_category_values: bool = False,
) -> dict[str, Any]:
    """
    The only view of a profile that is sent to an LLM.

    Keeps aggregate statistics; drops what is data rather than a statistic -
    observed category labels (`top_values`, `mode`, unless
    include_category_values=True) and the uploaded file's name - plus the
    LOCAL_ONLY_KEYS no issue uses. Size is bounded by max_columns and
    MAX_LLM_COLUMN_NAMES, never by row count.

    `columns` restricts the view to those columns (default: all of them).
    """
    drop = set(LOCAL_ONLY_KEYS) | (set() if include_category_values else set(CATEGORY_VALUE_KEYS))
    stats = report["columns"]
    selected = list(stats) if columns is None else [c for c in columns if c in stats]

    dataset = {k: v for k, v in report["dataset"].items() if k != "name"}
    names = dataset["column_names"]
    dataset["column_names"] = names[:MAX_LLM_COLUMN_NAMES]

    view: dict[str, Any] = {
        "dataset": dataset,
        "columns": {
            col: {k: v for k, v in stats[col].items() if k not in drop and v is not None}
            for col in selected[:max_columns]
        },
    }
    notes = []
    if len(selected) > max_columns:
        notes.append(f"Only the first {max_columns} of {len(selected)} columns are included.")
    if len(names) > MAX_LLM_COLUMN_NAMES:
        notes.append(f"column_names lists the first {MAX_LLM_COLUMN_NAMES} of {len(names)}.")
    if notes:
        view["note"] = " ".join(notes)
    return view


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def detect_delimiter(data: bytes) -> str:
    """
    The field separator, read off the header line. Excel writes ";" wherever
    the decimal separator is a comma, and such a file read with "," collapses
    into one column. Ties, and a header with no separator at all, mean ",".
    """
    header = data[:65536].split(b"\n", 1)[0]
    best = max(CSV_DELIMITERS, key=header.count)
    return best.decode() if header.count(best) else ","


def read_csv_bytes(
    data: bytes, max_rows: int | None = None, seed: int = 0
) -> tuple[pd.DataFrame, int, str]:
    """
    Parse CSV bytes. Returns (df, total_rows, encoding).

    Decoding falls back through CSV_ENCODINGS, so Excel's cp1252 exports load,
    and the delimiter (comma, semicolon, tab or pipe) is read off the header.
    When the file has more than max_rows data rows, a uniform random sample of
    max_rows is parsed and the parser skips the rest, so peak memory follows
    the sample rather than the file; total_rows is then estimated from the
    line count (quoted multi-line fields make it an overestimate).
    """
    lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
    total_rows = max(lines - 1, 0)  # minus the header
    sep = detect_delimiter(data)

    skiprows = None
    if max_rows is not None and total_rows > max_rows:
        rng = np.random.default_rng(seed)
        keep = set((rng.choice(total_rows, size=max_rows, replace=False) + 1).tolist())
        skiprows = lambda i: i > 0 and i not in keep  # noqa: E731 - row 0 is the header

    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding, skiprows=skiprows, sep=sep)
        except UnicodeDecodeError:
            continue
        return df, (total_rows if skiprows else len(df)), encoding
    raise AssertionError("unreachable: latin-1 decodes any byte sequence")


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
    python profiler.py                    -> profile the built-in demo dataset
    python profiler.py data.csv           -> profile your own CSV
    python profiler.py data.csv --json    -> dump the full (local) profile as JSON
    python profiler.py data.csv --payload -> dump the redacted view an LLM receives
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="AnalyseIt statistical profiler")
    parser.add_argument("csv", nargs="?", help="path to a CSV file (omit to use the demo dataset)")
    parser.add_argument("--json", action="store_true", help="print the full local profile as JSON")
    parser.add_argument("--payload", action="store_true",
                        help="print the redacted payload an LLM receives, as JSON")
    parser.add_argument("--target", help="label column to leave out of issue detection")
    parser.add_argument("--top", type=int, default=10, help="how many issues to list (default 10)")
    args = parser.parse_args()

    if args.csv:
        try:
            df, _, _ = read_csv_bytes(Path(args.csv).read_bytes())
        except FileNotFoundError:
            print(f"ERROR: no such file: {args.csv}")
            return 1
        except Exception as exc:
            print(f"ERROR: could not read {args.csv}: {exc}")
            return 1
        name = args.csv
    else:
        df, name = _demo_frame(), "demo.csv (built-in synthetic dataset)"

    try:
        profiler = DataProfiler(df, name=name, target=args.target)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    report = profiler.profile()

    if args.json:
        print(profiler.to_json())
        return 0
    if args.payload:
        print(json.dumps(profiler.to_llm_payload(), indent=2, default=str))
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
    print("  Run with --json for the full profile, --payload for the redacted LLM view.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
