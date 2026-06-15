"""
agent.py

The FitFindr planning loop. Orchestrates the three tools in response to a
natural language user query, passing state between them via a session dict.

The loop is CONDITIONAL: it branches on what search_listings returns. An
impossible query terminates early with session["error"] set and never reaches
the LLM tools; a matchable query flows through all three tools.

Usage (once implemented):
    from agent import run_agent
    from utils.data_loader import get_example_wardrobe

    result = run_agent(
        query="vintage graphic tee under $30, size M",
        wardrobe=get_example_wardrobe(),
    )
    print(result["fit_card"])
    print(result["error"])   # None on success
"""

import re

from tools import search_listings, suggest_outfit, create_fit_card


# ── session state ─────────────────────────────────────────────────────────────

def _new_session(query: str, wardrobe: dict) -> dict:
    """
    Initialize and return a fresh session dict for one user interaction.

    The session dict is the single source of truth for everything that happens
    during a run — it stores the original query, parsed parameters, tool results,
    and any error that caused early termination.
    """
    return {
        "query": query,              # original user query
        "parsed": {},                # extracted description / size / max_price
        "search_results": [],        # list of matching listing dicts
        "selected_item": None,       # top result, passed into suggest_outfit
        "wardrobe": wardrobe,        # user's wardrobe dict
        "outfit_suggestion": None,   # string returned by suggest_outfit
        "fit_card": None,            # string returned by create_fit_card
        "retry_note": None,          # set if the stretch retry loosened the search
        "error": None,               # set if the interaction ended early
    }


# ── query parsing ─────────────────────────────────────────────────────────────

# Word/phrase forms of clothing sizes -> normalized token.
_SIZE_WORDS = {
    "xx-small": "XXS", "xxs": "XXS",
    "x-small": "XS", "extra small": "XS", "xs": "XS",
    "small": "S",
    "medium": "M",
    "large": "L",
    "x-large": "XL", "extra large": "XL", "xl": "XL",
    "xx-large": "XXL", "xxl": "XXL",
}

# Cues that introduce the user's EXISTING wardrobe (not the item being searched).
# The search description is truncated before these so wardrobe talk doesn't pollute it.
_WARDROBE_CUES = [
    "i mostly wear", "i usually wear", "i normally wear", "i typically wear",
    "i mostly", "i usually", "i wear", "i own", "i have", "i've got", "ive got",
    "to pair", "to wear with", "to go with", "that goes with", "goes with",
    "pair it with", "to style with", "my wardrobe", "my closet",
]

# Conversational filler stripped from the parsed description.
_DESC_FILLER = {
    "im", "i'm", "i", "a", "an", "the", "looking", "look", "for", "some",
    "something", "want", "wanting", "need", "find", "show", "me", "get",
    "really", "just", "any", "kind", "of", "please", "hi", "hey", "searching",
    "in", "my", "size", "with", "to", "and", "or", "out", "there", "how",
    "would", "style", "it",
}


def _normalize_size(raw: str) -> str:
    raw = raw.strip()
    if raw in _SIZE_WORDS:
        return _SIZE_WORDS[raw]
    if raw.startswith("us"):
        return ("US " + raw[2:].strip()).strip()
    return raw.upper()


def _extract_price(q: str):
    """Return (max_price, matched_span) or (None, None)."""
    for pat in (
        r'(?:under|below|less than|cheaper than|no more than|max(?:imum)?|up to)\s*\$?\s*(\d+(?:\.\d+)?)',
        r'\$\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:dollars|usd|bucks)\b',
    ):
        m = re.search(pat, q)
        if m:
            return float(m.group(1)), m.span()
    return None, None


# Size token forms used by the extractors. Ordered so "xxs" wins over "xs", etc.
_SIZE_TOKEN = (
    r'us\s?\d+(?:\.\d+)?|w\d{2,3}(?:\s*l\d{2,3})?|\d+(?:\.\d+)?'
    r'|xxs|xx-small|xs|x-small|xxl|xx-large|xl|x-large|small|medium|large|[sml]'
)
# Explicit "size <X>" — accepts any size form including words and digits.
_SIZE_AFTER = re.compile(r'\bsize\s+(' + _SIZE_TOKEN + r')\b')
# Standalone size tokens that are UNAMBIGUOUS without the word "size"
# (W30, US 8, XS/XL/XXS/XXL). Bare digits and single S/M/L are intentionally
# excluded here to avoid false matches.
_SIZE_STANDALONE = re.compile(
    r'\b(us\s?\d+(?:\.\d+)?|w\d{2,3}(?:\s*l\d{2,3})?|xxs|xx-small|xs|x-small|xxl|xx-large|xl|x-large)\b'
)
# Determiner-led word sizes ("a small", "in a medium") — needs the article so
# "medium wash" / "light wash" / "large print" are NOT misread as a size filter.
_SIZE_DETERMINER = re.compile(r'\b(?:in\s+)?an?\s+(small|medium|large)\b')


def _extract_size(q: str):
    """Return (normalized_size, matched_span) or (None, None)."""
    m = _SIZE_AFTER.search(q)          # "size M", "size 8", "size US 8", "size medium"
    if m:
        return _normalize_size(m.group(1)), m.span()
    m = _SIZE_STANDALONE.search(q)     # "W30", "US 8", "XL" — unambiguous standalone
    if m:
        return _normalize_size(m.group(1)), m.span()
    m = _SIZE_DETERMINER.search(q)     # "in a medium", "a small"
    if m:
        return _normalize_size(m.group(1)), m.span()
    return None, None


def _parse_query(query: str) -> dict:
    """
    Parse a free-text query into {description, size, max_price}.

    Deterministic regex parsing (documented choice over an LLM parse: it's fast,
    free, and needs no API call). Price and size are extracted from the whole
    query; the search description is taken from the leading clause (before any
    sentence break or wardrobe cue) with price/size phrases and filler removed.
    """
    query = "" if query is None else str(query)
    q = query.lower().strip()
    max_price, price_span = _extract_price(q)
    size, size_span = _extract_size(q)

    # Description target = text before the first sentence end or wardrobe cue.
    cut = len(q)
    sm = re.search(r'[.!?]', q)
    if sm:
        cut = min(cut, sm.start())
    for cue in _WARDROBE_CUES:
        ci = q.find(cue)
        if ci != -1:
            cut = min(cut, ci)

    # Blank out the exact matched price/size spans (so they don't pollute the
    # description), then strip filler.
    chars = list(q[:cut])
    for span in (price_span, size_span):
        if span:
            for i in range(span[0], min(span[1], len(chars))):
                chars[i] = " "
    target = "".join(chars)

    target = re.sub(r"[^a-z0-9\s/'-]", " ", target)
    tokens = [t for t in target.split() if t not in _DESC_FILLER]
    description = " ".join(tokens).strip()
    if not description:
        description = query.strip()  # fallback: never search on an empty string

    return {"description": description, "size": size, "max_price": max_price}


def _no_results_message(parsed: dict) -> str:
    """Actionable error message naming what was tried and what to change."""
    tried = []
    if parsed.get("size"):
        tried.append(f"size {parsed['size']}")
    if parsed.get("max_price") is not None:
        tried.append(f"under ${parsed['max_price']:g}")
    constraints = (" " + " and ".join(tried)) if tried else ""
    return (
        f"I couldn't find any listings matching “{parsed['description']}”{constraints}. "
        "Try removing the size, raising your budget, or using broader keywords "
        "(for example “jacket” instead of a very specific style)."
    )


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Main agent entry point. Runs the FitFindr planning loop for a single
    user interaction and returns the completed session dict.

    Args:
        query:    Natural language user request
                  (e.g., "vintage graphic tee under $30, size M")
        wardrobe: User's wardrobe dict — use get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py

    Returns:
        The session dict after the interaction completes. Check session["error"]
        first — if it is not None, the interaction ended early and the other
        output fields (outfit_suggestion, fit_card) will be None.
    """
    # Step 1: fresh session — the single source of truth for this interaction.
    session = _new_session(query, wardrobe)

    # Step 2: parse the query -> description / size / max_price (written to session).
    session["parsed"] = _parse_query(session["query"])

    # Step 3: search, reading the parsed params back OUT of the session.
    session["search_results"] = search_listings(
        session["parsed"]["description"],
        session["parsed"]["size"],
        session["parsed"]["max_price"],
    )

    # Step 4: BRANCH on the search result held in the session.
    if not session["search_results"]:
        # Stretch — retry with the size filter loosened (only when one was applied).
        if session["parsed"]["size"] is not None:
            retry = search_listings(
                session["parsed"]["description"], None, session["parsed"]["max_price"]
            )
            if retry:
                session["retry_note"] = (
                    f"No matches in size {session['parsed']['size']}, so I dropped "
                    "the size filter and broadened the search."
                )
                session["search_results"] = retry

        if not session["search_results"]:
            # Required behavior: set error, leave LLM outputs None, return early.
            session["error"] = _no_results_message(session["parsed"])
            return session  # do NOT call suggest_outfit on an empty result set

    # Step 4b: select the top-ranked item — written once, read by both LLM tools.
    session["selected_item"] = session["search_results"][0]

    # Step 5: suggest an outfit, reading the selected item + wardrobe from the session.
    session["outfit_suggestion"] = suggest_outfit(
        session["selected_item"], session["wardrobe"]
    )

    # Step 6: turn the outfit into a shareable fit card — both inputs from the session.
    session["fit_card"] = create_fit_card(
        session["outfit_suggestion"], session["selected_item"]
    )

    # Step 7: done.
    return session


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe

    print("=== Happy path: graphic tee ===\n")
    session = run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    )
    print(f"parsed:        {session['parsed']}")
    print(f"# results:     {len(session['search_results'])}")
    if session["error"]:
        print(f"Error: {session['error']}")
    else:
        # State-flow proof: the SAME selected_item dict feeds both LLM tools.
        sel = session["selected_item"]
        print(f"selected_item: {sel['id']} — {sel['title']} (${sel['price']:g}, {sel['platform']})")
        print(f"\nOutfit:\n{session['outfit_suggestion']}")
        print(f"\nFit card:\n{session['fit_card']}")

    print("\n\n=== No-results path (must NOT call suggest_outfit) ===\n")
    session2 = run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    )
    print(f"parsed:            {session2['parsed']}")
    print(f"error:             {session2['error']}")
    print(f"selected_item:     {session2['selected_item']}")
    print(f"outfit_suggestion: {session2['outfit_suggestion']}")
    print(f"fit_card:          {session2['fit_card']}")
