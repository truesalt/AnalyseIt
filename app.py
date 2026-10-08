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
import os
import threading
import time
from collections import deque

import pandas as pd
import streamlit as st

from llm_agent import (
    DEFAULT_MODELS,
    LLMConfig,
    LLMError,
    build_script,
    default_config,
    generate_plan,
    openai_service_name,
    select_issues,
    unknown_columns,
)
from profiler import (
    MAX_LLM_COLUMNS,
    DataProfiler,
    detect_delimiter,
    llm_view,
    read_csv_bytes,
    with_target,
)
from rag_engine import KnowledgeBase

# Streamlit Community Cloud gives each app 1 GB of RAM. The embedding model and
# ChromaDB take ~600 MB of that, so the DataFrame must stay small.
MAX_PROFILE_ROWS = 250_000
MAX_UPLOAD_MB = 50

# The app's own key is shared by every visitor of a public deployment. A
# session counter alone resets on page refresh, hence the process-wide cap too.
HOST_KEY_RUNS_PER_SESSION = 5
HOST_KEY_RUNS_PER_HOUR = 60

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
def load_and_profile(file_bytes: bytes, filename: str) -> tuple[dict, int, bool, str]:
    """
    Parse the CSV and profile it. Cached on file content, so reruns are free.

    Returns (profile, total_rows, was_sampled, encoding). Files over
    MAX_PROFILE_ROWS are sampled while parsing, so the whole file is never
    held as a DataFrame.
    """
    df, total_rows, encoding = read_csv_bytes(file_bytes, max_rows=MAX_PROFILE_ROWS)
    sampled = total_rows > len(df)
    return DataProfiler(df, name=filename).profile(), total_rows, sampled, encoding


@st.cache_resource(show_spinner=False)
def _host_key_runs() -> tuple[threading.Lock, deque]:
    """Timestamps of shared-key runs across every session in this process."""
    return threading.Lock(), deque()


def claim_host_key_run() -> str | None:
    """Count one run on the shared key. Returns why it is refused, or None."""
    used = st.session_state.get("host_key_runs", 0)
    if used >= HOST_KEY_RUNS_PER_SESSION:
        return (
            f"This session has used its {HOST_KEY_RUNS_PER_SESSION} plans on the app's shared "
            "key. Paste your own key in the sidebar to continue."
        )
    lock, runs = _host_key_runs()
    now = time.time()
    with lock:
        while runs and now - runs[0] > 3600:
            runs.popleft()
        if len(runs) >= HOST_KEY_RUNS_PER_HOUR:
            return (
                "The app's shared key has reached its hourly limit. Paste your own key "
                "in the sidebar, or try again later."
            )
        runs.append(now)
    st.session_state["host_key_runs"] = used + 1
    return None


def release_host_key_run() -> None:
    """Give back a run that produced no plan (bad key, no credit, outage)."""
    st.session_state["host_key_runs"] = max(st.session_state.get("host_key_runs", 1) - 1, 0)
    lock, runs = _host_key_runs()
    with lock:
        if runs:
            runs.pop()


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

    # Start on ANALYSEIT_LLM_PROVIDER / ANALYSEIT_LLM_MODEL if set, else on the
    # provider that has a key. Reading st.secrets first copies its root-level
    # keys into the environment, which is all default_config() looks at.
    read_secret("ANTHROPIC_API_KEY")
    defaults = default_config()
    # OPENAI_BASE_URL can point the OpenAI option at Gemini or another
    # OpenAI-compatible API; label it with the service actually used.
    service = openai_service_name()
    openai_label = "OpenAI" if service == "OpenAI" else f"{service} (OpenAI-compatible)"
    provider = st.radio(
        "LLM provider",
        options=list(DEFAULT_MODELS.keys()),
        index=list(DEFAULT_MODELS).index(defaults.provider),
        format_func=lambda p: {"anthropic": "Anthropic (Claude)", "openai": openai_label}[p],
        horizontal=True,
    )
    model = st.text_input("Model", value=default_config(provider).model)

    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    host_key = read_secret(env_var)

    user_key = st.text_input(
        "Your API key (optional)",
        type="password",
        help=(
            "Used for this session only - never stored or logged. Leave blank to use "
            "the app's own key if one is configured."
        ),
        placeholder="sk-ant-..." if provider == "anthropic" else (
            "sk-..." if service == "OpenAI" else f"{service} API key"
        ),
    )

    api_key = user_key or host_key
    if user_key:
        st.caption("Using your key for this session.")
    elif host_key:
        st.caption("Using the app's shared key.")
    else:
        # Not an error: profiling needs no key, only plan generation does.
        st.info(
            f"No API key set. Profiling works without one; to generate a cleaning "
            f"plan, paste a key above or set {env_var}."
        )

    n_issues = st.slider(
        "Issues to analyse",
        1,
        5,
        2,
        help="The top issues, plus any other issues on the same columns, so each column "
        "gets one consistent fix.",
    )
    target_slot = st.container()  # filled once a CSV is loaded

    st.divider()
    kb_ready = False
    try:
        kb = get_knowledge_base()
        kb_ready = True
        st.caption(f"Knowledge base: {kb.count()} chunks · {kb.backend}")
    except Exception as exc:  # index build can fail on a cold container
        st.error(f"Knowledge base unavailable: {exc}")

    st.caption(
        "Your CSV stays in this process. Only aggregate statistics (counts, "
        "percentages, quantiles, skew) are sent to the LLM - never rows or category labels."
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
fid = hashlib.sha1(file_bytes).hexdigest()
size_mb = len(file_bytes) / 1024**2
if size_mb > MAX_UPLOAD_MB:
    st.error(f"File is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB.")
    st.stop()

try:
    with st.spinner("Profiling..."):
        profile, total_rows, was_sampled, encoding = load_and_profile(file_bytes, uploaded.name)
except pd.errors.EmptyDataError:
    st.error("That CSV appears to be empty.")
    st.stop()
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if profile["dataset"]["n_rows"] == 0:
    st.error("That CSV has a header row but no data rows.")
    st.stop()

target = target_slot.selectbox(
    "Target column (optional)",
    options=[None, *profile["columns"]],
    format_func=lambda c: "(none)" if c is None else c,
    help=(
        "The label you intend to predict. It is left out of issue detection: skew, "
        "scaling and encoding advice does not apply to a label."
    ),
)
if target is not None:
    profile = with_target(profile, target)

# A plan belongs to one file and one target. Drop it when either changes, or
# the previous dataset's code stays on screen under the new one's issues.
plan_key = f"{fid}:{target}"
if st.session_state.get("plan_key") != plan_key:
    st.session_state.pop("suggestions", None)
    st.session_state.pop("annotated", None)
    st.session_state["plan_key"] = plan_key

dataset = profile["dataset"]
issues = profile["issues"]

if was_sampled:
    st.warning(
        f"About {total_rows:,} rows exceeds the {MAX_PROFILE_ROWS:,}-row profiling limit. "
        f"Statistics below come from a random sample of {dataset['n_rows']:,} rows."
    )

# How the file was read - shown, and reproduced at the top of the downloaded plan.
sep = detect_delimiter(file_bytes)
read_notes = []
if sep != ",":
    read_notes.append({";": "semicolon", "\t": "tab", "|": "pipe"}[sep] + "-separated")
if encoding != "utf-8-sig":
    read_notes.append(f"{encoding}-encoded (it is not valid UTF-8)")
if read_notes:
    st.caption("Read as " + ", ".join(read_notes) + ".")
read_call = "pd.read_csv({name!r}{sep}{enc})".format(
    name=uploaded.name,
    sep=f", sep={sep!r}" if sep != "," else "",
    enc=f", encoding={encoding!r}" if encoding != "utf-8-sig" else "",
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{dataset['n_rows']:,}")
c2.metric("Columns", dataset["n_columns"])
c3.metric("Missing cells", f"{dataset['missing_cells_pct']}%")
c4.metric("Duplicate rows", f"{dataset['duplicate_rows']:,}")
if target is not None:
    st.caption(f"Target: `{target}` - left out of issue detection.")

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
            # The detail is the part worth reading; give it the room.
            column_config={
                "#": st.column_config.NumberColumn(width="small"),
                "Severity": st.column_config.TextColumn(width="small"),
                "Detail": st.column_config.TextColumn(width="large"),
            },
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
    st.caption(
        "What the LLM receives: this redacted view of the profile, with category labels "
        f"(`top_values`, `mode`) and the file name removed and at most {MAX_LLM_COLUMNS} "
        "columns. Each request carries only the slice for the column it is about."
    )
    st.json(llm_view(profile), expanded=2)  # dataset stats and column names, columns folded
    with st.expander("Full local profile (never sent)"):
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
    top = select_issues(issues, n_issues)
    config = LLMConfig(provider=provider, model=model or None, api_key=api_key)
    refusal = None if user_key else claim_host_key_run()

    if refusal:
        st.error(refusal)
    else:
        try:
            with st.spinner(f"Retrieving best practices for {len(top)} issue(s)..."):
                annotated = kb.retrieve_for_issues(top, top_k=3)
            with st.spinner(f"Generating code with {config.model}..."):
                suggestions = generate_plan(profile, annotated, config=config)
            st.session_state["suggestions"] = suggestions
            st.session_state["annotated"] = annotated
        except Exception as exc:
            if not user_key:
                release_host_key_run()
            st.error(str(exc) if isinstance(exc, LLMError) else f"Unexpected failure: {exc}")

if api_key and not user_key:
    left = HOST_KEY_RUNS_PER_SESSION - st.session_state.get("host_key_runs", 0)
    st.caption(f"Shared key: {max(left, 0)} of {HOST_KEY_RUNS_PER_SESSION} plans left in this session.")

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
            invented = unknown_columns(suggestion.python_code, dataset["column_names"])
            if invented:
                st.warning(
                    "Reads columns that are not in your data: "
                    + ", ".join(f"`{c}`" for c in invented)
                )
            if suggestion.is_safe_and_valid and not invented:
                st.success("Syntax valid, no unsafe constructs, and every column it reads exists.")

    with st.expander("Retrieved source passages"):
        for issue in st.session_state.get("annotated", []):
            st.markdown(f"**{issue['issue_type']}** — query: `{issue['rag_query']}`")
            if not issue.get("retrieved_context"):
                st.caption("No passage cleared the relevance cutoff.")
            for hit in issue.get("retrieved_context", []):
                st.caption(f"[{hit['similarity']:.3f}] {hit['section']}")
                st.text(hit["text"][:500])

    st.download_button(
        "Download plan as .py",
        data=build_script(
            suggestions, dataset["name"], known_columns=dataset["column_names"], read_call=read_call
        ),
        help="Steps whose code failed the syntax or safety check are commented out.",
        file_name=f"analyseit_plan_{fid[:8]}.py",
        mime="text/x-python",
        width="stretch",
    )
