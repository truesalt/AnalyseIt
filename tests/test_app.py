"""The Streamlit UI, driven headlessly with AppTest. Marked `model` because the
sidebar builds the knowledge base on start. The LLM is always stubbed."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.model

REPO = Path(__file__).resolve().parent.parent
SAMPLE = (REPO / "sample_messy.csv").read_bytes()
OTHER = b"x,y\n1,a\n2,b\n3,\n4,a\n5,b\n6,a\n7,b\n8,a\n9,b\n10,a\n"


def start() -> AppTest:
    at = AppTest.from_file(str(REPO / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    return at


def upload(at: AppTest, name: str, data: bytes) -> AppTest:
    at.file_uploader[0].set_value((name, data, "text/csv")).run()
    assert not at.exception
    return at


def plan_steps(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if m.value.startswith("### Step")]


def test_cold_start():
    at = start()
    assert at.info[0].value == "Upload a CSV to begin."
    assert any("Knowledge base: 36 chunks" in c.value for c in at.sidebar.caption)


def test_switching_provider_switches_model_and_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    at = start()
    at.sidebar.radio[0].set_value("openai").run()
    assert at.sidebar.text_input[0].value == "gpt-4o-mini"
    assert any("OPENAI_API_KEY" in w.value for w in at.sidebar.warning)


def test_sidebar_starts_on_the_provider_that_has_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    at = start()
    assert at.sidebar.radio[0].value == "openai"
    assert at.sidebar.text_input[0].value == "gpt-4o-mini"
    assert any("shared key" in c.value for c in at.sidebar.caption)


def test_upload_renders_issues_and_the_redacted_payload():
    at = upload(start(), "sample.csv", SAMPLE)
    assert at.tabs[0].label.startswith("Issues (")
    issues = at.dataframe[0].value
    assert ("customer_id", "identifier_like") in set(zip(issues["Column"], issues["Issue"]))
    assert "churned" not in set(issues["Column"])  # a 0/1 column is not "skewed"
    assert "top_values" not in at.json[0].value    # what the LLM receives
    assert "top_values" in at.json[1].value        # the full local profile


def test_new_upload_clears_the_previous_plan(stub_llm):
    at = upload(start(), "a.csv", SAMPLE)
    at.button[0].click().run()
    assert plan_steps(at) and stub_llm
    upload(at, "b.csv", OTHER)
    assert plan_steps(at) == []


def test_target_column_is_excluded_and_clears_the_plan(stub_llm):
    at = upload(start(), "a.csv", SAMPLE)
    at.button[0].click().run()
    assert "annual_income" in set(at.dataframe[0].value["Column"])
    at.sidebar.selectbox[0].set_value("annual_income").run()
    assert "annual_income" not in set(at.dataframe[0].value["Column"])
    assert plan_steps(at) == []


def test_shared_key_is_capped_per_session_but_own_key_is_not(stub_llm):
    at = upload(start(), "a.csv", SAMPLE)
    for _ in range(5):
        at.button[0].click().run()
    assert not at.error and "0 of 5 plans left" in " ".join(c.value for c in at.caption)
    at.button[0].click().run()
    assert any("used its 5 plans" in e.value for e in at.error)
    at.sidebar.text_input[1].set_value("my-own-key").run()  # bring your own key
    at.button[0].click().run()
    assert not at.error


def test_failed_run_does_not_use_up_a_shared_key_plan(monkeypatch):
    import llm_agent

    def no_credit(system, user, config):
        raise llm_agent.LLMError("Your OpenAI account has no credits left.")

    monkeypatch.setattr(llm_agent, "_call_anthropic", no_credit)
    at = upload(start(), "a.csv", SAMPLE)
    at.button[0].click().run()
    assert any("no credits" in e.value for e in at.error)
    assert "5 of 5 plans left" in " ".join(c.value for c in at.caption)


def test_invented_columns_are_flagged(monkeypatch):
    import llm_agent

    monkeypatch.setattr(llm_agent, "_call_anthropic", lambda system, user, config: llm_agent.CodeSuggestion(
        issue_detected="fix", theory_explanation="t", python_code="df['age'] = df['agee'].fillna(0)"))
    at = upload(start(), "a.csv", SAMPLE)
    at.button[0].click().run()
    assert any("`agee`" in w.value for w in at.warning)
    assert not at.success  # no green tick next to code that will raise a KeyError


def test_header_only_csv_is_reported():
    at = upload(start(), "empty.csv", b"a,b,c\n")
    assert any("no data rows" in e.value for e in at.error)


def test_excel_cp1252_csv_loads():
    at = upload(start(), "excel.csv", "city,price\nZürich,10\nMünchen,12\nSão Paulo,9\n".encode("cp1252"))
    assert any("cp1252" in c.value for c in at.caption)


def test_semicolon_csv_is_split_into_columns():
    at = upload(start(), "eu.csv", b"kunde;stadt;umsatz\n1;Berlin;12,5\n2;Koeln;7,25\n3;Berlin;9,75\n")
    assert at.metric[1].value == "3"  # columns, not one column holding every field
    assert any("semicolon-separated" in c.value for c in at.caption)
