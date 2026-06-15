"""
tools.py

The three required FitFindr tools (plus one stretch tool). Each tool is a
standalone function that can be called and tested independently before being
wired into the agent loop.

Tools:
    search_listings(description, size, max_price)  -> list[dict]   (pure, no LLM)
    suggest_outfit(new_item, wardrobe)             -> str          (LLM)
    create_fit_card(outfit, new_item)              -> str          (LLM)
    compare_price(item, listings=None)             -> str          (stretch, pure)

Every LLM call routes through the single _chat() helper, so the tools can be
unit-tested without a live API key by monkeypatching tools._chat.
"""

import os
import re

from dotenv import load_dotenv
from groq import Groq

from utils.data_loader import load_listings

load_dotenv()

# Groq's free-tier model for this project (same as Project 1).
_MODEL = "llama-3.3-70b-versatile"


# ── Groq client + chat helper ───────────────────────────────────────────────

def _get_groq_client():
    """Initialize and return a Groq client using GROQ_API_KEY from .env."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to a .env file in the project root."
        )
    return Groq(api_key=api_key)


def _chat(messages: list[dict], temperature: float = 0.7, max_tokens: int = 400) -> str:
    """
    Single choke point for every LLM call. Returns the model's text response.

    Centralizing the call here means the LLM-backed tools (suggest_outfit,
    create_fit_card) can be exercised in tests by monkeypatching tools._chat —
    no network or API key required for the unit tests.
    """
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


# ── prompt-formatting helpers ───────────────────────────────────────────────

def _format_item_for_prompt(item: dict) -> str:
    """Render a listing dict as a compact, readable block for an LLM prompt."""
    if not isinstance(item, dict):
        return str(item)
    parts = [
        f"- Title: {item.get('title', '?')}",
        f"- Category: {item.get('category', '?')}",
        f"- Style tags: {', '.join(str(t) for t in (item.get('style_tags') or []))}",
        f"- Colors: {', '.join(str(c) for c in (item.get('colors') or []))}",
        f"- Size: {item.get('size', '?')}",
        f"- Condition: {item.get('condition', '?')}",
        f"- Price: ${item.get('price', '?')} on {item.get('platform', '?')}",
    ]
    if item.get("description"):
        parts.append(f"- Description: {item['description']}")
    return "\n".join(parts)


def _format_wardrobe_for_prompt(items: list[dict]) -> str:
    """Render wardrobe items as a named, readable list for an LLM prompt."""
    lines = []
    for it in items:
        if not isinstance(it, dict):
            continue
        name = it.get("name", "item")
        cat = it.get("category", "")
        colors = ", ".join(str(c) for c in (it.get("colors") or []))
        tags = ", ".join(str(t) for t in (it.get("style_tags") or []))
        line = f"- {name} ({cat}; colors: {colors}; style: {tags}"
        if it.get("notes"):
            line += f"; note: {it['notes']}"
        line += ")"
        lines.append(line)
    return "\n".join(lines)


# ── search scoring helpers ──────────────────────────────────────────────────

# Grammar / query-filler words that carry no relevance signal.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "for", "with", "to", "in", "on", "at",
    "my", "me", "i", "im", "want", "wanting", "need", "looking", "look", "find",
    "show", "get", "some", "something", "under", "below", "over", "above",
    "around", "about", "size", "sized", "fits", "fit", "is", "are", "that",
    "this", "it", "its", "please", "up", "dollars", "dollar", "usd", "than",
    "less", "price", "priced", "budget", "wear", "wearing", "outfit", "piece",
    "pieces", "item", "items", "thrift", "thrifted", "secondhand",
}

# Field name -> weight applied when a query term hits that field.
_FIELD_WEIGHTS = [
    ("style_tags", 3),
    ("title", 3),
    ("category", 2),
    ("description", 1),
    ("colors", 1),
    ("brand", 1),
]

_CONDITION_RANK = {"excellent": 0, "good": 1, "fair": 2}


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on any non-alphanumeric run."""
    return [t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t]


def _size_tokens(size: str) -> set[str]:
    """Token set for a size string, e.g. 'US 8.5' -> {'us','8','5'}."""
    return set(_tokenize(size))


def _size_matches(query_size: str, listing_size: str) -> bool:
    """
    Token-set intersection match — robust to heterogeneous size formats.

    'M'  -> {m}   matches 'S/M' -> {s,m} and 'M/L' -> {m,l}
                  but NOT 'US 7' -> {us,7}  (avoids the 's'-in-'us' false match).
    '8'  -> {8}   matches 'US 8' and 'US 8.5'.
    """
    q = _size_tokens(query_size)
    if not q:
        return True  # no usable size constraint -> don't filter
    return bool(q & _size_tokens(listing_size))


def _query_terms(description: str) -> list[str]:
    """Meaningful query tokens (stopwords removed, single letters dropped)."""
    terms = []
    for t in _tokenize(description):
        if t in _STOPWORDS:
            continue
        if len(t) < 2 and not t.isdigit():
            continue
        terms.append(t)
    return terms


def _score_listing(terms: list[str], listing: dict) -> int:
    """Weighted keyword-overlap score for one listing against the query terms."""
    field_token_sets = {
        "style_tags": set(_tokenize(" ".join(listing.get("style_tags", []) or []))),
        "title": set(_tokenize(listing.get("title", ""))),
        "category": set(_tokenize(listing.get("category", ""))),
        "description": set(_tokenize(listing.get("description", ""))),
        "colors": set(_tokenize(" ".join(listing.get("colors", []) or []))),
        "brand": set(_tokenize(listing.get("brand", "") or "")),
    }
    score = 0
    for term in terms:
        for field, weight in _FIELD_WEIGHTS:
            if term in field_token_sets[field]:
                score += weight

    # Phrase bonus: adjacent query bigrams that appear verbatim in title/tags
    # (e.g. "graphic tee"). Caught beyond single-token overlap.
    phrase_haystack = (
        listing.get("title", "") + " " + " ".join(listing.get("style_tags", []) or [])
    ).lower()
    for a, b in zip(terms, terms[1:]):
        if f"{a} {b}" in phrase_haystack:
            score += 3
    return score


# ── Tool 1: search_listings ──────────────────────────────────────────────────

def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the mock listings dataset for items matching the description,
    optional size, and optional price ceiling.

    Args:
        description: Keywords describing what the user is looking for.
        size:        Size string to filter by, or None to skip size filtering.
                     Matching is case-insensitive and token-based
                     (e.g., "M" matches "S/M" but not "US 7").
        max_price:   Maximum price (inclusive), or None to skip price filtering.

    Returns:
        A list of matching listing dicts, sorted by relevance (best match first).
        Returns an empty list if nothing matches — does NOT raise an exception.
        Each dict has the dataset fields: id, title, description, category,
        style_tags, size, condition, price, colors, brand, platform.
    """
    # Defensive coercion so the tool never raises on odd caller input.
    description = "" if description is None else str(description)
    if size is not None:
        size = str(size)
    if max_price is not None and not isinstance(max_price, (int, float)):
        try:
            max_price = float(max_price)
        except (TypeError, ValueError):
            max_price = None

    listings = load_listings()

    # 1) hard filters: price ceiling, then size.
    filtered = []
    for item in listings:
        if not isinstance(item, dict):
            continue
        if max_price is not None:
            price = item.get("price")
            if not isinstance(price, (int, float)) or price > max_price:
                continue
        if size is not None and not _size_matches(size, item.get("size", "")):
            continue
        filtered.append(item)

    # 2) relevance scoring on the survivors.
    terms = _query_terms(description)
    scored = []
    for item in filtered:
        score = _score_listing(terms, item)
        if score > 0:
            scored.append((score, item))

    # 3) sort: score desc, then condition (excellent>good>fair), price asc, id.
    def _sort_price(item):
        p = item.get("price")
        return p if isinstance(p, (int, float)) else float("inf")

    scored.sort(
        key=lambda si: (
            -si[0],
            _CONDITION_RANK.get(si[1].get("condition", ""), 3),
            _sort_price(si[1]),
            str(si[1].get("id", "")),
        )
    )
    return [item for _, item in scored]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest 1–2 complete outfits.

    Args:
        new_item: A listing dict (the item the user is considering buying).
        wardrobe: A wardrobe dict with an 'items' key containing a list of
                  wardrobe item dicts. May be empty — handled gracefully.

    Returns:
        A non-empty string with outfit suggestions. If the wardrobe is empty,
        returns general styling advice for the item rather than naming items
        the user does not own. Never raises — returns a graceful descriptive
        string if the LLM call fails.
    """
    try:
        item_block = _format_item_for_prompt(new_item)
        items = wardrobe.get("items") if isinstance(wardrobe, dict) else None
        items = items if isinstance(items, list) else []
        if not items:
            # Empty / new-user wardrobe -> general styling advice.
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are FitFindr, a friendly secondhand-fashion stylist. "
                        "Keep advice concrete and practical in 3-5 sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "A shopper is considering this thrifted item:\n"
                        f"{item_block}\n\n"
                        "They haven't told me what's in their closet yet. Give GENERAL "
                        "styling advice: what kinds of pieces (colors, silhouettes, "
                        "shoes) pair well with it, what vibe or occasion it suits, and "
                        "one specific outfit idea they could build around it. Do NOT "
                        "invent specific items they own."
                    ),
                },
            ]
        else:
            wardrobe_block = _format_wardrobe_for_prompt(items)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are FitFindr, a friendly secondhand-fashion stylist. "
                        "Suggest 1-2 complete outfits that pair the NEW item with "
                        "pieces the shopper ALREADY OWNS. Reference the owned pieces "
                        "by name. Be concrete in 3-6 sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"New thrifted item:\n{item_block}\n\n"
                        f"The shopper's current wardrobe:\n{wardrobe_block}\n\n"
                        "Suggest 1-2 complete outfit combinations that pair the new "
                        "item with specific pieces from their wardrobe (name them). "
                        "Mention shoes and a layering or accessory option where it fits."
                    ),
                },
            ]
        result = _chat(messages, temperature=0.7, max_tokens=400)
        if result and result.strip():
            return result
        err = None  # model returned nothing usable
    except Exception as exc:  # never crash the agent on an API/LLM error
        err = type(exc).__name__

    # Fallback for an empty model response or an API error — always non-empty.
    title = new_item.get("title", "this piece") if isinstance(new_item, dict) else "this piece"
    prefix = f"(Heads up — I couldn't reach the styling model right now: {err}.) " if err else ""
    return (
        prefix
        + f"As a quick idea, '{title}' pairs well with simple basics in neutral colors: "
        "try well-fitting denim and clean sneakers or boots, then add one layer "
        "(a jacket or overshirt) to finish the look."
    )


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Generate a short, shareable outfit caption for the thrifted find.

    Args:
        outfit:   The outfit suggestion string from suggest_outfit().
        new_item: The listing dict for the thrifted item.

    Returns:
        A 2–4 sentence Instagram/TikTok-style caption string. If outfit is empty
        or whitespace-only, returns a descriptive error message string WITHOUT
        calling the LLM. Never raises — returns a graceful descriptive string if
        the LLM call fails. Uses a high temperature so captions vary across runs.
    """
    # Guard FIRST — no LLM call when there's nothing (or nothing valid) to caption.
    if not isinstance(outfit, str) or not outfit.strip():
        return (
            "Can't write a fit card without an outfit suggestion yet — run "
            "suggest_outfit first to describe the look, then I'll caption it."
        )

    is_dict = isinstance(new_item, dict)
    title = new_item.get("title", "this piece") if is_dict else "this piece"
    price = new_item.get("price") if is_dict else None
    platform = (new_item.get("platform") if is_dict else "") or "a resale app"
    price_str = f"${price:g}" if isinstance(price, (int, float)) else "a steal"

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are FitFindr. Write ONE short, casual, authentic "
                    "Instagram/TikTok-style OOTD caption (2-4 sentences). It should "
                    "sound like a real person sharing a thrift find, NOT a product "
                    "description. Mention the item name, its price, and the platform "
                    "naturally — once each. Capture the outfit's specific vibe and "
                    "vary your wording. Lowercase-casual is fine; 0-2 emojis are ok."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Item: {title}\n"
                    f"Price: {price_str}\n"
                    f"Platform: {platform}\n"
                    f"Outfit: {outfit}\n\n"
                    "Write the caption."
                ),
            },
        ]
        # High temperature -> different wording for different inputs and runs.
        result = _chat(messages, temperature=1.0, max_tokens=180)
        if result and result.strip():
            return result
        err = None  # model returned nothing usable
    except Exception as exc:  # never crash the agent on an API/LLM error
        err = type(exc).__name__

    # Fallback for an empty model response or an API error — always non-empty.
    prefix = f"(Couldn't reach the caption model right now: {err}.) " if err else ""
    return (
        prefix
        + f"thrifted the {title} for {price_str} on {platform} and styling it "
        "exactly like the look above ✨"
    )


# ── Tool 4 (stretch): compare_price ───────────────────────────────────────────

def compare_price(item: dict, listings: list[dict] | None = None) -> str:
    """
    Estimate whether an item's price is fair versus comparable listings.

    Args:
        item:     A listing dict with at least 'price' and 'category'.
        listings: Pool of comparables; defaults to the full dataset.

    Returns:
        A short verdict string comparing the item's price to the average price
        of same-category comparables (e.g. "below market"/"around market"/
        "above market" with the numbers). Returns a descriptive string (never
        raises) when there isn't enough data to compare.
    """
    if not isinstance(item, dict) or not isinstance(item.get("price"), (int, float)):
        return "Can't compare price — the item is missing a numeric price."

    pool = listings if listings is not None else load_listings()
    category = item.get("category")
    price = float(item["price"])

    comps = [
        l for l in pool
        if l.get("category") == category
        and isinstance(l.get("price"), (int, float))
        and l.get("id") != item.get("id")
    ]
    if not comps:
        return (
            f"Not enough comparable {category or 'similar'} listings to judge "
            f"whether ${price:g} is a fair price."
        )

    avg = sum(float(l["price"]) for l in comps) / len(comps)
    diff_pct = (price - avg) / avg * 100 if avg else 0.0

    if diff_pct <= -15:
        verdict = "a good deal — below market"
    elif diff_pct >= 15:
        verdict = "on the pricey side — above market"
    else:
        verdict = "about right — around market"

    return (
        f"At ${price:g}, this is {verdict} for {category}: comparable {category} "
        f"listings average ${avg:.2f} (across {len(comps)} items)."
    )
