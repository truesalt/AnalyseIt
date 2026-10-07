"""
llm_agent.py - AnalyseIt's brain: strictly typed LLM code generation.

Takes the profiler's statistics JSON plus the passages retrieved from the
knowledge base and returns a validated CodeSuggestion.

THE CORE CONSTRAINT: the LLM never sees raw dataset rows. Only aggregate
statistics and retrieved textbook text are sent. `_reject_raw_data()` enforces
this at the API boundary - passing a DataFrame raises rather than silently
uploading someone's data.

Providers (either works; same Pydantic object comes back):
    anthropic  ->  claude-haiku-4-5    via client.messages.parse()
    openai     ->  gpt-4o-mini         via client.chat.completions.parse()

Usage:
    suggestion = generate_cleaning_code(stats_json, retrieved_context)
    print(suggestion.python_code)
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

try:  # optional - loads OPENAI_API_KEY / ANTHROPIC_API_KEY from a .env file
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
}

# max_tokens is a ceiling, not a charge - only generated tokens are billed, so
# keep it generous. A truncated response fails Pydantic validation entirely.
DEFAULT_MAX_TOKENS = 16000

# Patterns that must never appear in generated code. The app only *displays*
# code, but a user may paste it straight into a notebook.
FORBIDDEN_CODE_PATTERNS = [
    (r"\bimport\s+(os|sys|subprocess|shutil|socket|requests|urllib)\b", "system/network import"),
    (r"\bfrom\s+(os|sys|subprocess|shutil|socket|requests|urllib)\b", "system/network import"),
    (r"\b(eval|exec|compile)\s*\(", "dynamic code execution"),
    (r"\b__import__\s*\(", "dynamic import"),
    (r"\bopen\s*\(", "file I/O"),
    (r"\b(to_csv|to_pickle|read_csv|read_pickle|to_sql)\s*\(", "file/database I/O"),
    (r"\bpickle\b", "pickle"),
]


class LLMError(RuntimeError):
    """Raised when the provider call fails or returns something unusable."""


# Sampling parameters (temperature/top_p/top_k) were removed on Claude 4.6 and
# later - sending temperature to those models returns a 400. Haiku 4.5 and
# earlier still accept it, but only via extra_body (see _call_anthropic).
# Keeps a model swap from breaking the call.
_NO_TEMPERATURE_PATTERNS = (
    "claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
    "claude-sonnet-4-6", "claude-sonnet-5", "claude-fable", "claude-mythos",
)


def _supports_temperature(provider: str, model: str) -> bool:
    if provider != "anthropic":
        return True
    return not any(model.startswith(p) for p in _NO_TEMPERATURE_PATTERNS)


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------
class CodeSuggestion(BaseModel):
    """One data-quality issue, the theory behind the fix, and runnable code."""

    issue_detected: str = Field(
        description="The specific data quality issue found, naming the affected column(s)."
    )
    theory_explanation: str = Field(
        description=(
            "Why this issue matters and why the chosen fix is correct, grounded in the "
            "retrieved reference material. 2-4 sentences."
        )
    )
    python_code: str = Field(
        description=(
            "Runnable pandas/scikit-learn code operating on a DataFrame named `df`. "
            "No imports of os/sys/subprocess, no file or network I/O, no plotting."
        )
    )
    columns_affected: list[str] = Field(
        default_factory=list, description="Column names this code modifies."
    )
    execution_order: int = Field(
        default=1,
        description=(
            "Position in a full cleaning pipeline: 1 drop empty/constant, 2 deduplicate, "
            "3 fix dtypes, 4 impute, 5 outliers, 6 transform skew, 7 encode, 8 scale."
        ),
    )

    @property
    def syntax_error(self) -> str | None:
        """None when python_code parses; otherwise the SyntaxError message."""
        try:
            ast.parse(self.python_code)
            return None
        except SyntaxError as exc:
            return f"line {exc.lineno}: {exc.msg}"

    @property
    def safety_warnings(self) -> list[str]:
        """Forbidden constructs found in the generated code."""
        return [
            label
            for pattern, label in FORBIDDEN_CODE_PATTERNS
            if re.search(pattern, self.python_code)
        ]

    @property
    def is_safe_and_valid(self) -> bool:
        return self.syntax_error is None and not self.safety_warnings


@dataclass
class LLMConfig:
    """Provider settings. api_key=None falls back to the environment."""

    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str | None = None
    api_key: str | None = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0  # deterministic: same stats should give the same fix
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.provider not in DEFAULT_MODELS:
            raise ValueError(
                f"Unknown provider {self.provider!r}; expected 'anthropic' or 'openai'."
            )
        self.model = self.model or DEFAULT_MODELS[self.provider]

    def resolve_key(self) -> str:
        env_var = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        key = self.api_key or os.getenv(env_var)
        if not key:
            raise LLMError(
                f"No API key for {self.provider}. Set {env_var} in the environment or a "
                f".env file, or pass api_key= explicitly."
            )
        return key


def default_config() -> LLMConfig:
    """Provider from ANALYSEIT_LLM_PROVIDER, else whichever key is present."""
    provider = os.getenv("ANALYSEIT_LLM_PROVIDER", "").strip().lower()
    if provider not in DEFAULT_MODELS:
        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else "openai"
    return LLMConfig(
        provider=provider,  # type: ignore[arg-type]
        model=os.getenv("ANALYSEIT_LLM_MODEL") or None,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a senior data scientist generating data cleaning code for an automated EDA tool.

WHAT YOU RECEIVE
You are given (1) a JSON summary of *aggregate statistics* about a dataset and
(2) excerpts from a machine learning reference text. You never see the raw rows,
and you must not ask for them.

RULES
1. Work only from the statistics provided. Never invent a column name, a dtype,
   or a value that does not appear in the JSON. If the statistics are
   insufficient to justify a fix, say so in theory_explanation.
2. Ground theory_explanation in the reference excerpts provided. Where the
   excerpts give a threshold or a rule, apply that rule rather than your own.
3. python_code must be runnable pandas / scikit-learn that operates on an
   existing DataFrame named `df` and modifies it in place or reassigns it.
   - Include any needed imports for pandas, numpy, or sklearn only.
   - Never import os, sys, subprocess, shutil, socket, requests, or urllib.
   - Never read or write files, never call eval/exec, never plot.
   - Reference columns by the exact names in the JSON, quoted as strings.
   - Add brief inline comments explaining each step.
4. Where a fix learns a statistic from the data (an imputation value, a scaler,
   an encoder), note in a comment that it must be fitted on the training split
   only, to avoid leakage.
5. Keep the code focused on the issue described. Do not produce a whole pipeline
   unless the issue is about pipeline ordering.
"""

USER_TEMPLATE = """\
## Dataset statistics (aggregate only - no raw rows)

```json
{stats_json}
```

## Reference material retrieved from the knowledge base

{context}

## Task

{task}

Produce one focused fix: name the issue, explain the reasoning using the
reference material above, and give runnable pandas code that operates on `df`.
"""


def _reject_raw_data(stats: Any) -> None:
    """
    Enforce the project's core constraint at the API boundary.

    A DataFrame or Series reaching this function would mean raw rows are about
    to be serialized into a prompt and sent to a third party. Fail loudly.
    """
    if hasattr(stats, "iloc") and hasattr(stats, "to_dict"):
        raise LLMError(
            "Refusing to send a pandas object to the LLM. AnalyseIt sends aggregate "
            "statistics only - pass DataProfiler(df).profile() or .to_llm_payload()."
        )


def build_prompt(
    stats_json: dict[str, Any] | str,
    retrieved_context: str | list[dict[str, Any]],
    task: str | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt). Separated out so it is unit-testable."""
    _reject_raw_data(stats_json)

    if isinstance(stats_json, str):
        stats_text = stats_json
    else:
        stats_text = json.dumps(stats_json, indent=2, default=str)

    if isinstance(retrieved_context, str):
        context_text = retrieved_context
    else:
        # A list of hits from rag_engine.retrieve_context().
        context_text = "\n\n".join(
            f"[Source {i} | {hit.get('section', 'reference')}]\n{hit.get('text', '')}"
            for i, hit in enumerate(retrieved_context, start=1)
        )
    context_text = context_text.strip() or "(no reference material retrieved)"

    task = task or (
        "Identify the single most important data quality issue in these statistics "
        "and write the code to fix it."
    )
    return SYSTEM_PROMPT, USER_TEMPLATE.format(
        stats_json=stats_text, context=context_text, task=task
    )


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------
def _call_anthropic(system: str, user: str, config: LLMConfig) -> CodeSuggestion:
    import anthropic

    client = anthropic.Anthropic(api_key=config.resolve_key())
    kwargs: dict[str, Any] = dict(config.extra)
    if _supports_temperature(config.provider, config.model or ""):
        # anthropic 1.x removed temperature from the messages.* signatures, so
        # passing it by name is a TypeError regardless of model. The models that
        # still accept it take it through the request body instead.
        extra_body = dict(kwargs.pop("extra_body", {}))
        extra_body.setdefault("temperature", config.temperature)
        kwargs["extra_body"] = extra_body
    try:
        response = client.messages.parse(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=CodeSuggestion,
            **kwargs,
        )
    except anthropic.AuthenticationError as exc:
        raise LLMError("Anthropic rejected the API key.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("Anthropic rate limit hit. Wait and retry.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Could not reach the Anthropic API. Check connectivity.") from exc

    if response.stop_reason == "max_tokens":
        raise LLMError("Response was truncated; raise max_tokens.")
    parsed = response.parsed_output
    if parsed is None:
        raise LLMError("Anthropic returned no parseable structured output.")
    return parsed


def _call_openai(system: str, user: str, config: LLMConfig) -> CodeSuggestion:
    import openai

    client = openai.OpenAI(api_key=config.resolve_key())
    try:
        completion = client.chat.completions.parse(
            model=config.model,
            max_completion_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=CodeSuggestion,
            **config.extra,
        )
    except openai.AuthenticationError as exc:
        raise LLMError("OpenAI rejected the API key.") from exc
    except openai.RateLimitError as exc:
        raise LLMError("OpenAI rate limit or quota exceeded.") from exc
    except openai.APIStatusError as exc:
        raise LLMError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
    except openai.APIConnectionError as exc:
        raise LLMError("Could not reach the OpenAI API. Check connectivity.") from exc

    choice = completion.choices[0]
    if choice.finish_reason == "length":
        raise LLMError("Response was truncated; raise max_tokens.")
    if getattr(choice.message, "refusal", None):
        raise LLMError(f"OpenAI refused the request: {choice.message.refusal}")
    parsed = choice.message.parsed
    if parsed is None:
        raise LLMError("OpenAI returned no parseable structured output.")
    return parsed


# A public deployment re-analyses the same demo datasets constantly; an
# identical prompt should not be billed twice.
_CACHE: dict[str, CodeSuggestion] = {}


def generate_cleaning_code(
    stats_json: dict[str, Any] | str,
    retrieved_context: str | list[dict[str, Any]],
    config: LLMConfig | None = None,
    task: str | None = None,
    use_cache: bool = True,
) -> CodeSuggestion:
    """
    Generate one validated CodeSuggestion from statistics + retrieved context.

    Raises LLMError on any provider failure, so callers handle one exception type.
    """
    config = config or default_config()
    system, user = build_prompt(stats_json, retrieved_context, task)

    cache_key = hashlib.sha256(
        f"{config.provider}|{config.model}|{system}|{user}".encode("utf-8")
    ).hexdigest()
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

    if config.provider == "anthropic":
        suggestion = _call_anthropic(system, user, config)
    else:
        suggestion = _call_openai(system, user, config)

    if use_cache:
        _CACHE[cache_key] = suggestion
    return suggestion


def generate_plan(
    profile: dict[str, Any],
    annotated_issues: list[dict[str, Any]],
    config: LLMConfig | None = None,
) -> list[CodeSuggestion]:
    """
    Generate one suggestion per issue, then order them into a runnable sequence.

    Takes the output of KnowledgeBase.retrieve_for_issues(). Issues on the same
    column are sent in a single request so the model cannot propose a log
    transform and an outlier clip that contradict each other - `income` firing
    both high_skew and outliers from one lognormal tail is the common case.
    """
    by_column: dict[str, list[dict[str, Any]]] = {}
    for issue in annotated_issues:
        by_column.setdefault(str(issue.get("column")), []).append(issue)

    suggestions: list[CodeSuggestion] = []
    for column, issues in by_column.items():
        context: list[dict[str, Any]] = []
        seen_sections = set()
        for issue in issues:
            for hit in issue.get("retrieved_context", []):
                key = (hit.get("section"), hit.get("text", "")[:60])
                if key not in seen_sections:
                    seen_sections.add(key)
                    context.append(hit)

        described = "; ".join(f"{i['issue_type']} - {i['detail']}" for i in issues)
        scope = f"column '{column}'" if column != "None" else "the dataset as a whole"
        task = (
            f"These issues were detected on {scope}: {described}\n\n"
            f"Write ONE combined fix that addresses them together without the steps "
            f"contradicting each other."
        )

        stats_slice = {
            "dataset": profile["dataset"],
            "columns": (
                {column: profile["columns"][column]}
                if column in profile.get("columns", {})
                else profile.get("columns", {})
            ),
        }
        suggestions.append(
            generate_cleaning_code(stats_slice, context, config=config, task=task)
        )

    suggestions.sort(key=lambda s: s.execution_order)
    return suggestions


# ---------------------------------------------------------------------------
# CLI - offline prompt inspection, plus a live call when a key is present
# ---------------------------------------------------------------------------
def _cli() -> int:
    """
    python llm_agent.py --dry-run     build and print the prompt, no API call
    python llm_agent.py               make a real call (needs an API key)
    """
    import argparse
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import pandas as pd

    from profiler import DataProfiler
    from rag_engine import KnowledgeBase

    parser = argparse.ArgumentParser(description="AnalyseIt LLM agent")
    parser.add_argument("--dry-run", action="store_true", help="print the prompt, do not call the API")
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--model")
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "customer_id": range(400),
            "income": rng.lognormal(10, 1.2, 400).round(2),
            "city": rng.choice(["Delhi", "Mumbai", "Pune", None], 400, p=[0.4, 0.3, 0.2, 0.1]),
        }
    )
    df.loc[rng.choice(400, 130, replace=False), "income"] = np.nan

    profiler = DataProfiler(df, name="demo.csv")
    kb = KnowledgeBase()
    kb.ensure_index()
    annotated = kb.retrieve_for_issues(profiler.top_issues(2), top_k=2)

    payload = profiler.to_llm_payload()
    context = annotated[0]["retrieved_context"]
    system, user = build_prompt(payload, context, task=annotated[0]["detail"])

    if args.dry_run:
        print("=" * 78 + "\nSYSTEM PROMPT\n" + "=" * 78)
        print(system)
        print("=" * 78 + "\nUSER PROMPT\n" + "=" * 78)
        print(user)
        print("=" * 78)
        print(f"prompt chars: {len(system) + len(user):,} (~{(len(system) + len(user)) // 4:,} tokens)")
        print(f"raw rows included: 0")
        return 0

    config = LLMConfig(provider=args.provider or default_config().provider, model=args.model)
    print(f"provider={config.provider} model={config.model}\n")
    try:
        suggestion = generate_cleaning_code(payload, context, config=config)
    except LLMError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"ISSUE      : {suggestion.issue_detected}")
    print(f"COLUMNS    : {suggestion.columns_affected}")
    print(f"ORDER      : {suggestion.execution_order}")
    print(f"\nTHEORY     : {suggestion.theory_explanation}")
    print(f"\nCODE:\n{'-' * 78}\n{suggestion.python_code}\n{'-' * 78}")
    print(f"syntax ok  : {suggestion.syntax_error is None}")
    print(f"warnings   : {suggestion.safety_warnings or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
