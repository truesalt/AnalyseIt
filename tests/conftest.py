"""Shared fixtures. No test reaches a real LLM: provider calls are stubbed or
served by a mock HTTP transport, and the API keys are dummies."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import llm_agent

REPO = Path(__file__).resolve().parent.parent
SAMPLE_CSV = REPO / "sample_messy.csv"


@pytest.fixture(autouse=True)
def _isolated_llm_state(monkeypatch):
    """Dummy keys (so nothing can bill a real account) and an empty response cache."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("ANALYSEIT_EMBED_BACKEND", "onnx")
    llm_agent._CACHE.clear()
    yield
    llm_agent._CACHE.clear()


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def stub_llm(monkeypatch):
    """Replace both provider calls with a recorder. Returns the list of user prompts."""
    sent: list[str] = []

    def fake_call(system: str, user: str, config: llm_agent.LLMConfig) -> llm_agent.CodeSuggestion:
        sent.append(user)
        return llm_agent.CodeSuggestion(
            issue_detected=f"stub fix {len(sent)}",
            theory_explanation="stub",
            python_code="df = df.copy()",
            execution_order=len(sent),
        )

    monkeypatch.setattr(llm_agent, "_call_anthropic", fake_call)
    monkeypatch.setattr(llm_agent, "_call_openai", fake_call)
    return sent


@pytest.fixture(scope="session")
def kb():
    """In-memory ONNX knowledge base, built once per test session."""
    from rag_engine import KnowledgeBase

    knowledge_base = KnowledgeBase(backend="onnx", in_memory=True, collection_name="test_kb")
    knowledge_base.build_index()
    return knowledge_base
