"""
tests/test_agent.py

Tests for the planning loop and query parser. The LLM tools are stubbed by
monkeypatching agent.suggest_outfit / agent.create_fit_card, so these run
without a live API key and let us assert the loop's CONTROL FLOW directly
(state passing + the no-results early-return rule).

Run from the repo root:
    pytest tests/
"""

import agent
from agent import run_agent, _parse_query
from utils.data_loader import get_example_wardrobe


# ── query parser ─────────────────────────────────────────────────────────────

def test_parse_strips_price_and_filler():
    p = _parse_query("looking for a vintage graphic tee under $30")
    assert p["description"] == "vintage graphic tee"
    assert p["max_price"] == 30.0
    assert p["size"] is None


def test_parse_size_after_keyword():
    assert _parse_query("90s track jacket in size M")["size"] == "M"
    assert _parse_query("black combat boots size 8")["size"] == "8"
    assert _parse_query("designer ballgown size XXS under $5")["size"] == "XXS"


def test_parse_standalone_sizes():
    assert _parse_query("baggy jeans W30")["size"] == "W30"
    assert _parse_query("black combat boots US 8")["size"] == "US 8"
    assert _parse_query("track jacket in a medium")["size"] == "M"


def test_parse_medium_wash_is_not_a_size():
    # Regression: "medium wash" is a descriptor, not a size filter.
    p = _parse_query("vintage levi's medium wash jeans")
    assert p["size"] is None
    assert "medium" in p["description"]  # kept for search relevance


def test_parse_excludes_wardrobe_context():
    p = _parse_query(
        "vintage graphic tee under $30. i mostly wear baggy jeans and chunky sneakers"
    )
    assert p["description"] == "vintage graphic tee"
    assert "jeans" not in p["description"]


# ── planning loop ────────────────────────────────────────────────────────────

def _stub_llm(monkeypatch):
    monkeypatch.setattr(agent, "suggest_outfit", lambda item, wardrobe: f"OUTFIT::{item['id']}")
    monkeypatch.setattr(agent, "create_fit_card", lambda outfit, item: f"CARD::{item['id']}::{outfit}")


def test_run_agent_happy_path_state_flows_through_session(monkeypatch):
    _stub_llm(monkeypatch)
    s = run_agent("vintage graphic tee under $30", get_example_wardrobe())
    assert s["error"] is None
    # selected_item is the SAME object as the top search result (no copy/re-entry).
    assert s["selected_item"] is s["search_results"][0]
    # outfit + fit card were produced from exactly that selected item.
    sel_id = s["selected_item"]["id"]
    assert s["outfit_suggestion"] == f"OUTFIT::{sel_id}"
    assert s["fit_card"] == f"CARD::{sel_id}::OUTFIT::{sel_id}"


def test_run_agent_no_results_never_calls_suggest_outfit(monkeypatch):
    called = {"suggest": False}

    def spy_suggest(item, wardrobe):
        called["suggest"] = True
        return "should never run"

    monkeypatch.setattr(agent, "suggest_outfit", spy_suggest)
    monkeypatch.setattr(agent, "create_fit_card", lambda o, i: "should never run")

    s = run_agent("designer ballgown size XXS under $5", get_example_wardrobe())
    assert s["error"] is not None             # actionable error set
    assert s["selected_item"] is None
    assert s["outfit_suggestion"] is None
    assert s["fit_card"] is None              # left None per the hard rule
    assert called["suggest"] is False         # suggest_outfit NOT called on empty results


def test_run_agent_none_query_is_graceful():
    # Defensive: a None/blank query must not crash — it returns an error session.
    s = run_agent(None, get_example_wardrobe())
    assert s["error"] is not None
    assert s["selected_item"] is None
    assert s["fit_card"] is None


def test_run_agent_retry_fallback_loosens_size(monkeypatch):
    _stub_llm(monkeypatch)
    # No listing exists in size XXL, but "graphic tee" does -> retry drops the size.
    s = run_agent("graphic tee size XXL", get_example_wardrobe())
    assert s["error"] is None
    assert s["retry_note"] is not None        # told the user it broadened the search
    assert s["selected_item"] is not None
    assert s["fit_card"] is not None
