"""
tests/test_tools.py

Unit tests for the three FitFindr tools, with at least one test per failure
mode. The LLM-backed tools (suggest_outfit, create_fit_card) are tested by
monkeypatching tools._chat, so the whole suite runs WITHOUT a live API key.

Run from the repo root:
    pytest tests/
"""

import pytest

import tools
from tools import search_listings, suggest_outfit, create_fit_card, compare_price
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── search_listings: happy path + failure mode (no results) + filters ────────

def test_search_returns_results():
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    assert isinstance(results, list)
    assert len(results) > 0
    # Every result is a full listing dict with the documented fields.
    for r in results:
        for field in ("id", "title", "price", "category", "style_tags", "size"):
            assert field in r


def test_search_results_sorted_by_relevance():
    # The literal "Graphic Tee" title should out-rank a vintage graphic *hoodie*.
    results = search_listings("vintage graphic tee", size=None, max_price=50)
    titles = [r["title"] for r in results]
    assert any("Tee" in t for t in titles)
    # Top result is a tee, not the hoodie.
    assert "Hoodie" not in results[0]["title"]


def test_search_empty_results():
    # Failure mode: nothing matches -> empty list, NOT an exception.
    results = search_listings("designer ballgown", size="XXS", max_price=5)
    assert results == []


def test_search_price_filter():
    # Spec example: nothing in the dataset is <= $10, so this is [] (filter works).
    results = search_listings("jacket", size=None, max_price=10)
    assert all(item["price"] <= 10 for item in results)


def test_search_price_filter_excludes_pricier_items():
    # Stronger price-filter check: a realistic ceiling returns matches, all <= ceiling.
    results = search_listings("denim", size=None, max_price=30)
    assert len(results) > 0
    assert all(item["price"] <= 30 for item in results)
    # The $42 cropped denim jacket and $75 leather bomber must be excluded.
    ids = {item["id"] for item in results}
    assert "lst_007" not in ids
    assert "lst_022" not in ids


def test_search_size_filter_matches_token():
    # "M" should match listings sized "M", "S/M", "M/L" — case-insensitive token match.
    results = search_listings("vintage", size="M", max_price=None)
    assert len(results) > 0
    for item in results:
        sz = item["size"].lower()
        assert ("m" in sz.replace("us", ""))  # has an 'm' token outside "us"


def test_search_size_filter_no_false_positive():
    # "S" must NOT match shoe sizes like "US 7"/"US 8" (the 's' in 'us').
    results = search_listings("sneakers", size="S", max_price=None)
    for item in results:
        assert item["size"] not in ("US 7", "US 8", "US 8.5", "US 9")


def test_size_matches_unit():
    # Direct check of the token-set size matcher (intent, not dataset coincidence).
    assert tools._size_matches("M", "S/M") is True
    assert tools._size_matches("M", "M/L") is True
    assert tools._size_matches("M", "M") is True
    assert tools._size_matches("S", "US 7") is False   # the 's' in 'us' must not match
    assert tools._size_matches("8", "US 8.5") is True
    assert tools._size_matches("XL", "XL (oversized)") is True
    assert tools._size_matches("M", "W30 L30") is False


# ── suggest_outfit: empty-wardrobe failure mode + branch behavior ────────────

def _capture_chat(monkeypatch):
    """Replace tools._chat with a recorder; returns the captured-call list."""
    calls = []

    def fake_chat(messages, temperature=0.7, max_tokens=400):
        calls.append({"messages": messages, "temperature": temperature})
        return "MOCK_LLM_RESPONSE"

    monkeypatch.setattr(tools, "_chat", fake_chat)
    return calls


def test_suggest_outfit_empty_wardrobe(monkeypatch):
    # Failure mode: empty wardrobe -> general advice, non-empty string, no crash.
    calls = _capture_chat(monkeypatch)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = suggest_outfit(item, get_empty_wardrobe())
    assert isinstance(out, str) and out.strip()
    # Took the GENERAL-advice branch (prompt tells the model not to invent owned items).
    assert len(calls) == 1
    assert "GENERAL" in calls[0]["messages"][1]["content"]


def test_suggest_outfit_with_wardrobe_names_owned_pieces(monkeypatch):
    calls = _capture_chat(monkeypatch)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = suggest_outfit(item, get_example_wardrobe())
    assert isinstance(out, str) and out.strip()
    # Took the wardrobe branch: the prompt includes named owned pieces.
    prompt = calls[0]["messages"][1]["content"]
    assert "Baggy straight-leg jeans" in prompt


def test_suggest_outfit_handles_none_wardrobe(monkeypatch):
    _capture_chat(monkeypatch)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = suggest_outfit(item, None)  # defensive: None wardrobe must not crash
    assert isinstance(out, str) and out.strip()


def test_suggest_outfit_api_error_is_graceful(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated API outage")

    monkeypatch.setattr(tools, "_chat", boom)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = suggest_outfit(item, get_empty_wardrobe())
    assert isinstance(out, str) and out.strip()  # graceful fallback, no exception


def test_suggest_outfit_empty_response_is_graceful(monkeypatch):
    # Model returns nothing -> must still return a non-empty string (spec: never empty).
    monkeypatch.setattr(tools, "_chat", lambda *a, **k: "")
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = suggest_outfit(item, get_example_wardrobe())
    assert isinstance(out, str) and out.strip()


# ── create_fit_card: empty-outfit failure mode + variance mechanism ──────────

def test_create_fit_card_empty_outfit_no_llm_call(monkeypatch):
    # Failure mode: empty outfit -> descriptive error string, and NO LLM call.
    def boom(*a, **k):
        raise AssertionError("LLM must NOT be called when outfit is empty")

    monkeypatch.setattr(tools, "_chat", boom)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = create_fit_card("", item)
    assert isinstance(out, str) and out.strip()
    assert "suggest_outfit" in out  # descriptive guidance


def test_create_fit_card_whitespace_outfit_guarded(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must NOT be called for whitespace outfit")

    monkeypatch.setattr(tools, "_chat", boom)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = create_fit_card("   \n  ", item)
    assert isinstance(out, str) and out.strip()


def test_create_fit_card_uses_high_temperature_and_item_details(monkeypatch):
    calls = _capture_chat(monkeypatch)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = create_fit_card("Pair it with baggy jeans and chunky sneakers.", item)
    assert isinstance(out, str) and out.strip()
    # Variance mechanism: high temperature is wired in.
    assert calls[0]["temperature"] >= 0.9
    # Prompt carries the item name, price, and platform so they can be mentioned.
    prompt = calls[0]["messages"][1]["content"]
    assert item["title"] in prompt
    assert item["platform"] in prompt
    assert f"${item['price']:g}" in prompt


def test_create_fit_card_api_error_is_graceful(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated API outage")

    monkeypatch.setattr(tools, "_chat", boom)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = create_fit_card("Pair it with baggy jeans.", item)
    assert isinstance(out, str) and out.strip()  # graceful fallback, no exception


def test_create_fit_card_empty_response_is_graceful(monkeypatch):
    # Model returns whitespace -> must still return a non-empty caption string.
    monkeypatch.setattr(tools, "_chat", lambda *a, **k: "   ")
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = create_fit_card("Pair it with baggy jeans.", item)
    assert isinstance(out, str) and out.strip()


# ── compare_price (stretch) ──────────────────────────────────────────────────

def test_compare_price_returns_verdict():
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    verdict = compare_price(item)
    assert isinstance(verdict, str) and verdict.strip()
    assert "market" in verdict.lower()


def test_compare_price_missing_price_is_graceful():
    out = compare_price({"category": "tops"})  # no numeric price
    assert isinstance(out, str) and "Can't compare" in out


# ── robustness against odd/malformed inputs (tools must never raise) ─────────

def test_search_tolerates_odd_types():
    # Non-str description, int size, str max_price -> still returns a list, no raise.
    assert isinstance(search_listings(123, size=5, max_price="50"), list)
    assert isinstance(search_listings("tee", size=None, max_price=-1), list)
    assert isinstance(search_listings("", size="", max_price=None), list)


def test_create_fit_card_non_string_outfit_guarded(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("LLM must NOT be called for a non-string outfit")

    monkeypatch.setattr(tools, "_chat", boom)
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = create_fit_card(123, item)  # not a string
    assert isinstance(out, str) and out.strip()


def test_suggest_outfit_non_dict_wardrobe_is_graceful(monkeypatch):
    monkeypatch.setattr(tools, "_chat", lambda *a, **k: "MOCK")
    item = search_listings("vintage graphic tee", size=None, max_price=50)[0]
    out = suggest_outfit(item, "closet")  # wardrobe not a dict
    assert isinstance(out, str) and out.strip()


def test_suggest_outfit_malformed_item_is_graceful(monkeypatch):
    monkeypatch.setattr(tools, "_chat", lambda *a, **k: "MOCK")
    # style_tags with non-string elements must not crash prompt formatting.
    out = suggest_outfit({"title": "x", "style_tags": [1, 2]}, {"items": []})
    assert isinstance(out, str) and out.strip()
