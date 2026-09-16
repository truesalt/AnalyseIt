"""
app.py - AnalyseIt Streamlit interface.

Upload a CSV -> deterministic statistical profiling -> RAG lookup of ML best
practices -> LLM-generated cleaning code.

The uploaded CSV never leaves this process. Only the aggregate statistics
produced by DataProfiler are sent to the LLM.

Run:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os

import pandas as pd
import streamlit as st

from llm_agent import DEFAULT_MODELS, LLMConfig, LLMError, generate_plan
from profiler import DataProfiler
from rag_engine import KnowledgeBase

# Streamlit Community Cloud gives each app 1 GB of RAM. The embedding model and
# ChromaDB take ~600 MB of that, so the DataFrame must stay small.
MAX_PROFILE_ROWS = 250_000
MAX_UPLOAD_MB = 50

st.set_page_config(page_title="AnalyseIt", page_icon="🔍", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_knowledge_base() -> KnowledgeBase:
    """Built once per container, not per rerun. First call downloads the model."""
    kb = KnowledgeBase()
    kb.ensure_index()
    return kb


@st.cache_data(show_spinner=False, max_entries=4)
def load_and_profile(file_bytes: bytes, filename: str) -> tuple[dict, int, bool]:
    """
    Parse the CSV and profile it. Cached on file content, so reruns are free.

    Returns (profile, original_row_count, was_sampled).
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    original_rows = len(df)
    sampled = original_rows > MAX_PROFILE_ROWS
    if sampled:
        df = df.sample(MAX_PROFILE_ROWS, random_state=0)
    return DataProfiler(df, name=filename).profile(), original_rows, sampled


def read_secret(name: str) -> str | None:
    """st.secrets raises when no secrets.toml exists locally - fall back to env."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name)


# ---------------------------------------------------------------------------
# Sidebar: provider + key
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    provider = st.radio(
        "LLM provider",
        options=list(DEFAULT_MODELS.keys()),
        format_func=lambda p: {"anthropic": "Anthropic (Claude)", "openai": "OpenAI"}[p],
        horizontal=True,
    )
    model = st.text_input("Model", value=DEFAULT_MODELS[provider])

    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    host_key = read_secret(env_var)

    user_key = st.text_input(
        "Your API key (optional)",
        type="password",
        help=(
            "Used for this session only - never stored or logged. Leave blank to use "
            "the app's own key if one is configured."
        ),
        placeholder="sk-..." if provider == "openai" else "sk-ant-...",
    )

    api_key = user_key or host_key
    if user_key:
        st.caption("Using your key for this session.")
    elif host_key:
        st.caption("Using the app's shared key.")
    else:
        st.warning(f"No key available. Set {env_var} or paste one above.")

    n_issues = st.slider("Issues to analyse", 1, 5, 2)

    st.divider()
    kb_ready = False
    try:
        kb = get_knowledge_base()
        kb_ready = True
        st.caption(f"Knowledge base: {kb.count()} chunks · {kb.backend}")
    except Exception as exc:  # index build can fail on a cold container
        st.error(f"Knowledge base unavailable: {exc}")

    st.caption(
        "Your CSV stays in this process. Only aggregate statistics "
        "(counts, percentages, skew) are sent to the LLM."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("🔍 AnalyseIt")
st.caption(
    "Doc-augmented automated EDA and feature engineering. "
    "Statistical profiling finds the issues, a vector database supplies the theory, "
    "and the LLM writes the code — without ever seeing your data."
)

uploaded = st.file_uploader(
    "Upload a CSV", type=["csv"], help=f"Up to {MAX_UPLOAD_MB} MB."
)

if uploaded is None:
    st.info("Upload a CSV to begin.")
    with st.expander("How it works"):
        st.markdown(
            """
            1. **Profile** — `profiler.py` computes per-column statistics: missingness,
               cardinality, skewness, IQR and Z-score outliers, dtype mismatches — and
               ranks the resulting issues by severity.
            2. **Retrieve** — each issue carries a query that `rag_engine.py` runs against
               a local ChromaDB collection of ML best practices.
            3. **Generate** — `llm_agent.py` sends the statistics plus the retrieved
               passages to the LLM, which returns a validated `CodeSuggestion`.

            The LLM receives no rows — only aggregates such as `skewness: 5.61` and
            `missing_pct: 40.4`. That holds regardless of dataset size.
            """
        )
    st.stop()

file_bytes = uploaded.getvalue()
size_mb = len(file_bytes) / 1024**2
if size_mb > MAX_UPLOAD_MB:
    st.error(f"File is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.")
    st.stop()

try:
    with st.spinner("Profiling..."):
        profile, original_rows, was_sampled = load_and_profile(file_bytes, uploaded.name)
except pd.errors.EmptyDataError:
    st.error("That CSV appears to be empty.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

dataset = profile["dataset"]
issues = profile["issues"]

if was_sampled:
    st.warning(
        f"{original_rows:,} rows exceeds the {MAX_PROFILE_ROWS:,}-row profiling limit. "
        f"Statistics below come from a random sample of {MAX_PROFILE_ROWS:,} rows."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{dataset['n_rows']:,}")
c2.metric("Columns", dataset["n_columns"])
c3.metric("Missing cells", f"{dataset['missing_cells_pct']}%")
c4.metric("Duplicate rows", f"{dataset['duplicate_rows']:,}")

tab_issues, tab_columns, tab_json = st.tabs(
    [f"Issues ({len(issues)})", "Columns", "Profile JSON"]
)

with tab_issues:
    if not issues:
        st.success("No data quality issues detected.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "#": i["rank"],
                        "Severity": i["severity"],
                        "Column": i["column"] or "(dataset)",
                        "Issue": i["issue_type"],
                        "Detail": i["detail"],
                    }
                    for i in issues
                ]
            ),
            hide_index=True,
            width="stretch",
        )

with tab_columns:
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Column": name,
                    "Type": stats["semantic_type"],
                    "dtype": stats["dtype"],
                    "Missing %": stats["missing_pct"],
                    "Unique": stats["unique_count"],
                    "Skew": stats.get("skewness"),
                    "Outliers": stats.get("iqr_outlier_count"),
                }
                for name, stats in profile["columns"].items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )

with tab_json:
    st.caption("This is the structure sent to the LLM — no rows, only aggregates.")
    st.json(profile, expanded=False)


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
st.divider()

can_generate = bool(issues) and kb_ready and bool(api_key)
if not issues:
    st.info("Nothing to fix — no plan needed.")
elif not api_key:
    st.info("Add an API key in the sidebar to generate a plan.")

if st.button(
    "Generate Feature Engineering Plan",
    type="primary",
    disabled=not can_generate,
    width="stretch",
):
    top = issues[:n_issues]
    config = LLMConfig(provider=provider, model=model or None, api_key=api_key)

    try:
        with st.spinner(f"Retrieving best practices for {len(top)} issue(s)..."):
            annotated = kb.retrieve_for_issues(top, top_k=3)
        with st.spinner(f"Generating code with {config.model}..."):
            suggestions = generate_plan(profile, annotated, config=config)
        st.session_state["suggestions"] = suggestions
        st.session_state["annotated"] = annotated
    except LLMError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Unexpected failure: {exc}")

suggestions = st.session_state.get("suggestions")
if suggestions:
    st.subheader("Cleaning plan")
    st.caption("Steps are ordered so earlier fixes do not invalidate later ones.")

    for n, suggestion in enumerate(suggestions, start=1):
        with st.container(border=True):
            st.markdown(f"### Step {n} — {suggestion.issue_detected}")
            if suggestion.columns_affected:
                st.caption("Columns: " + ", ".join(f"`{c}`" for c in suggestion.columns_affected))

            st.markdown("**Why this fix**")
            st.write(suggestion.theory_explanation)

            st.markdown("**Generated code**")
            st.code(suggestion.python_code, language="python")

            if suggestion.syntax_error:
                st.error(f"Generated code does not parse — {suggestion.syntax_error}")
            if suggestion.safety_warnings:
                st.warning(
                    "Flagged constructs: " + ", ".join(suggestion.safety_warnings)
                )
            if suggestion.is_safe_and_valid:
                st.success("Syntax valid, no unsafe constructs.")

    with st.expander("Retrieved source passages"):
        for issue in st.session_state.get("annotated", []):
            st.markdown(f"**{issue['issue_type']}** — query: `{issue['rag_query']}`")
            for hit in issue.get("retrieved_context", []):
                st.caption(f"[{hit['similarity']:.3f}] {hit['section']}")
                st.text(hit["text"][:500])

    script = "\n\n".join(
        f"# Step {n}: {s.issue_detected}\n"
        f"# {s.theory_explanation.replace(chr(10), ' ')}\n{s.python_code}"
        for n, s in enumerate(suggestions, start=1)
    )
    st.download_button(
        "Download plan as .py",
        data=f"# AnalyseIt cleaning plan for {dataset['name']}\n"
        f"# Assumes a DataFrame named `df` is already loaded.\n\n{script}\n",
        file_name=f"analyseit_plan_{hashlib.sha1(file_bytes).hexdigest()[:8]}.py",
        mime="text/x-python",
        width="stretch",
    )
