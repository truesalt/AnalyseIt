"""Knowledge base: chunking (offline) and retrieval (needs the embedding model)."""

from __future__ import annotations

import pytest

from rag_engine import DEFAULT_DOCS_FILE, chunk_document


# --- chunking ---------------------------------------------------------------------
def test_preamble_before_the_first_section_is_skipped():
    chunks = chunk_document("# Title\n# a comment\n\n## Section A\n\nFirst sentence. Second sentence.\n")
    assert [c["section"] for c in chunks] == ["Section A"]
    assert chunks[0]["text"] == "Section A\n\nFirst sentence. Second sentence."


def test_sub_headings_and_hash_lines_inside_a_section_are_kept():
    text = (
        "## Imputation\n\nIntro sentence here.\n\n### Numeric columns\n\n"
        "Use the median.\n# fit on the training split only\nThen transform the test split.\n"
    )
    body = chunk_document(text)[0]["text"]
    assert "Numeric columns." in body
    assert "# fit on the training split only" in body


def test_chroma_telemetry_is_off():
    from rag_engine import KnowledgeBase

    assert KnowledgeBase(backend="onnx", in_memory=True).client.get_settings().anonymized_telemetry is False


def test_text_without_sections_yields_no_chunks():
    assert chunk_document("# just a title\nsome text\n") == []


def test_bundled_docs_chunk_cleanly():
    chunks = chunk_document(DEFAULT_DOCS_FILE.read_text(encoding="utf-8"))
    assert (len(chunks), len({c["section"] for c in chunks})) == (36, 18)
    for chunk in chunks:
        assert chunk["text"].startswith(chunk["section"] + "\n\n")
        assert chunk["text"].rstrip().endswith((".", "!", "?", '"', ")"))


# --- retrieval --------------------------------------------------------------------
@pytest.mark.model
def test_relevant_query_retrieves_its_section(kb):
    hits = kb.retrieve_context("how to handle columns that are completely empty", top_k=3)
    assert hits and hits[0]["section"] == "Handling columns that are completely empty"


@pytest.mark.model
def test_off_topic_query_retrieves_nothing(kb):
    assert kb.retrieve_context("best pizza toppings in naples") == []
    assert len(kb.retrieve_context("best pizza toppings in naples", top_k=3, min_similarity=0.0)) == 3


@pytest.mark.model
def test_retrieval_eval_passes():
    from eval_retrieval import run_eval

    report = run_eval(backend="onnx")
    assert report["top1_correct"] == len(report["queries"]) == 21
    assert report["issue_types_covered"] == 18
    assert report["passed"], report
