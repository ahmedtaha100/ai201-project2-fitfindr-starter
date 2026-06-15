# FitFindr planning.md

> Written before implementation (Milestone 2) and updated before each stretch feature.
> This spec locks down the design before implementation. The more specific the
> inputs, outputs, and branches are, the less the final code has to guess.

---

## Tools

At least 3 tools, each with a clearly defined interface. The three required tools
are below; two stretch tools follow in **Additional Tools**.

### Tool 1: search_listings

**What it does:**
Searches the 40 mock secondhand listings for items that match a free-text
description, optionally narrowed by size and a maximum price. Returns the matches
ranked by relevance. This is the only non-LLM tool — it is pure, deterministic
filtering + keyword scoring over `data/listings.json`.

**Input parameters:**
- `description` (str): free-text keywords describing the desired item, e.g.
  `"vintage graphic tee"`. Required.
- `size` (str | None): a size token to filter on, e.g. `"M"`, `"8"`, `"W30"`.
  `None` skips size filtering. Matching is case-insensitive and token-based.
- `max_price` (float | None): inclusive price ceiling in dollars. `None` skips
  price filtering.

**What it returns:**
`list[dict]` — the matching listing dicts, sorted by relevance score (best first).
Each dict has exactly the dataset fields: `id` (str), `title` (str),
`description` (str), `category` (str), `style_tags` (list[str]), `size` (str),
`condition` (str: excellent|good|fair), `price` (float), `colors` (list[str]),
`brand` (str|None), `platform` (str: depop|thredUp|poshmark). Returns an **empty
list `[]`** when nothing matches — never raises.

**How it ranks (relevance):**
1. Load all listings with `load_listings()`.
2. Drop listings above `max_price` (when given).
3. Drop listings whose size does not match (when `size` given) — token-set
   intersection: split both sizes on non-alphanumerics, lowercase; keep if the
   token sets share a token. (`"M"`→{m} matches `"S/M"`→{s,m} but not `"US 7"`→{us,7}.)
4. Score each survivor by keyword overlap of `description` tokens against the
   listing: `style_tags`×3, `title`×3, `category`×2, `description`×1, `colors`×1,
   `brand`×1, plus a +3 bonus when a query bigram (e.g. `"graphic tee"`) appears
   verbatim in title/tags.
5. Drop any listing scoring 0.
6. Sort by score desc; deterministic tie-break: condition rank (excellent>good>fair),
   then price ascending, then id.

**What happens if it fails or returns nothing:**
Returns `[]` (no exception). The planning loop detects the empty list and sets an
actionable `session["error"]` telling the user what to try (loosen the size, raise
the price, or use different keywords) and **stops before calling `suggest_outfit`**.
(Stretch retry-with-fallback may auto-retry with the size filter dropped first.)

---

### Tool 2: suggest_outfit

**What it does:**
Given one listing the user is considering and their current wardrobe, asks the LLM
(Groq `llama-3.3-70b-versatile`) to propose one or more complete outfit
combinations that pair the new item with named pieces the user already owns.

**Input parameters:**
- `new_item` (dict): a single listing dict (the item under consideration) — the
  same shape `search_listings` returns. Required.
- `wardrobe` (dict): a wardrobe dict with an `"items"` key holding a list of
  wardrobe-item dicts (`id`, `name`, `category`, `colors`, `style_tags`, `notes`).
  May be empty (`items == []`).

**What it returns:**
`str` — a non-empty, human-readable suggestion. With a populated wardrobe it names
specific owned pieces ("pair with your baggy straight-leg jeans and chunky white
sneakers…"). With an empty wardrobe it returns **general styling advice** for the
item (what kinds of pieces pair well, what vibe it suits) instead of naming
nonexistent items.

**What happens if it fails or returns nothing:**
- Empty wardrobe → general-advice branch (still useful, never empty).
- LLM/API error → caught; returns a graceful descriptive string (e.g. "Couldn't
  reach the styling model right now — here's a generic pairing idea: …"). Never raises.

---

### Tool 3: create_fit_card

**What it does:**
Turns an outfit suggestion + the new item into a short, shareable Instagram/TikTok
-style caption. Calls the LLM with a high temperature so the caption reads
differently for different inputs (and across runs on the same input).

**Input parameters:**
- `outfit` (str): the outfit suggestion text from `suggest_outfit()`. Required.
- `new_item` (dict): the listing dict, used to mention the item name, price, and
  platform naturally (once each).

**What it returns:**
`str` — a 2–4 sentence casual caption (not a product description) that captures the
outfit vibe and mentions the item name, price, and platform once each.

**What happens if it fails or returns nothing:**
- `outfit` empty / whitespace-only → returns a **descriptive error string**
  ("Can't write a fit card without an outfit suggestion — run suggest_outfit
  first.") **before** any LLM call. Never raises.
- LLM/API error → caught; returns a graceful descriptive string. Never raises.

---

### Additional Tools (stretch)

#### Tool 4 (stretch): compare_price
**What it does:** Estimates whether a listing's price is fair versus comparable
listings (same `category`) in the dataset.
**Inputs:** `item` (dict, the listing), `listings` (list[dict] | None — defaults to
`load_listings()`).
**Returns:** `str` — a verdict like "below market (avg $X for bottoms)" /
"around market" / "above market", with the comparison number. No comparables →
descriptive "not enough comparable listings" string. Never raises.
**Failure:** empty/invalid item or no comparables → descriptive string, not an exception.

#### Behavior 5 (stretch): retry-with-fallback (in the planning loop, not a standalone tool)
If `search_listings` returns `[]` **and** a `size` filter was applied, the loop
auto-retries once with `size=None`. If the retry finds matches, it records
`session["retry_note"]` ("No exact size-M matches, so I dropped the size filter and
found these instead.") and continues the happy path. Only if the retry is also
empty does it set `session["error"]`.

---

## Planning Loop

**How the agent decides which tool to call next (conditional, not a fixed sequence):**

1. **Initialize** the session with `_new_session(query, wardrobe)`.
2. **Parse** the natural-language `query` into `description`, `size`, `max_price`:
   - `max_price`: regex for `under $30` / `$30` / `30 dollars` / `below 30`.
   - `size`: regex for `size <X>` or a standalone size token (`S/M/L/XL/XS`,
     `W##`, `US ##`, a bare number after "size"); normalize `small/medium/large`→`S/M/L`.
   - `description`: the query with the matched price/size phrases and filler words
     removed (falls back to the whole query if stripping leaves it empty).
   - Store all three in `session["parsed"]`.
3. **Search:** call `search_listings(description, size, max_price)`; store in
   `session["search_results"]`.
4. **BRANCH on the result** (this is the required conditional behavior):
   - **Required — empty result:** if `search_results == []`, set `session["error"]`
     to an actionable message, leave `selected_item`, `outfit_suggestion`, and
     `fit_card` as `None`, and **return the session early — do NOT call
     `suggest_outfit`**. The base loop always honors this: the LLM tools are never
     reached on an empty result set.
   - **Optional stretch (retry-with-fallback, sanctioned by the spec's stretch
     list):** *before* falling into the empty-result behavior above, if a `size`
     filter was applied the loop re-runs `search_listings` once with `size=None`.
     If the loosened search returns matches, it records `session["retry_note"]` and
     continues down the **non-empty** path; if the loosened search is still empty it
     falls through to the required empty-result behavior. Because the retry only
     continues when results become non-empty, `suggest_outfit` is still **never**
     called on an empty result set — the stretch refines *how hard we look*, not the
     hard rule.
   - **Non-empty result:** `session["selected_item"] = search_results[0]` (top relevance).
5. **Suggest:** call `suggest_outfit(selected_item, wardrobe)`; store in
   `session["outfit_suggestion"]`.
6. **Fit card:** call `create_fit_card(outfit_suggestion, selected_item)`; store in
   `session["fit_card"]`.
7. **Return** the session.

The behavior visibly differs by input: an impossible query terminates at step 4
with an error and no LLM calls; a matchable query flows through all three tools.

---

## State Management

**How information from one tool flows to the next:**

A single `session` dict (created by `_new_session`) is the only source of truth for
one interaction. Each step **writes** its output to a named key and the next step
**reads** from that key — nothing is re-entered by the user or hardcoded.

| Key | Written by | Read by |
|-----|-----------|---------|
| `query` | `_new_session` | parser |
| `parsed` (`description`,`size`,`max_price`) | step 2 parser | `search_listings` (step 3) |
| `search_results` | `search_listings` | branch logic (step 4) |
| `selected_item` | step 4 (`= search_results[0]`) | `suggest_outfit` (step 5) **and** `create_fit_card` (step 6) |
| `wardrobe` | `_new_session` | `suggest_outfit` |
| `outfit_suggestion` | `suggest_outfit` | `create_fit_card` |
| `fit_card` | `create_fit_card` | UI |
| `error` | step 4 on no-results | UI (gates the other panels) |
| `retry_note` (stretch) | step 4 retry | UI |

The exact dict in `selected_item` is the exact dict passed into `suggest_outfit`
and `create_fit_card` — verified by printing it in Milestone 4.

---

## Error Handling

Each tool owns its failure mode; the loop surfaces it to the user and either falls
back or asks for more info — never fails silently, never crashes.

| Tool | Failure mode | Agent response |
|------|-------------|----------------|
| search_listings | No results match the query | Returns `[]`. Loop (stretch) auto-retries with the size filter dropped; if that finds items it tells the user "no exact size matches, so I broadened the search." If still nothing: `session["error"]` = "No listings matched 'designer ballgown' in size XXS under $5. Try removing the size, raising your budget, or using broader keywords like 'dress'." No `suggest_outfit` call. |
| suggest_outfit | Wardrobe is empty (new user) | Detects `wardrobe["items"] == []` and returns general styling advice for the item (vibe, what kinds of pieces pair well) instead of naming nonexistent items. On LLM/API error, returns a graceful descriptive fallback string. |
| create_fit_card | Outfit input missing/incomplete | Guards empty/whitespace `outfit` first and returns "Can't write a fit card without an outfit suggestion — run suggest_outfit first." (no LLM call). On LLM/API error, returns a graceful descriptive fallback string. |

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
   User query  ──────────►│                 run_agent()                  │
   + wardrobe choice      │               (PLANNING LOOP)                │
                          └──────────────────────────────────────────────┘
                                   │
                                   ▼
                         Step 2: parse query  ──► Session["parsed"] = {description, size, max_price}
                                   │
                                   ▼
                         Step 3: search_listings(description, size, max_price)
                                   │
                                   ▼  Session["search_results"]
                         ┌───────── BRANCH on results ──────────┐
            results == [] (and size filtered? → retry once)     results = [item, ...]
                                   │                                   │
              still empty │       ▼                                   ▼
                          ▼  Session["retry_note"]            Session["selected_item"] = results[0]
            ┌─────────────────────────────┐                          │
            │ [ERROR BRANCH]              │                          ▼
            │ Session["error"] = "No      │              Step 5: suggest_outfit(selected_item, wardrobe)
            │   listings matched… try…"   │                          │
            │ outfit_suggestion = None    │                          ▼  Session["outfit_suggestion"]
            │ fit_card = None             │              Step 6: create_fit_card(outfit_suggestion,
            │ ── return session early ──► │                                       selected_item)
            └─────────────────────────────┘                          │
                          │                                          ▼  Session["fit_card"]
                          │                                          │
                          └────────────►  return session  ◄──────────┘
                                                │
                                                ▼
                              app.py handle_query() maps session → 3 panels
                       (error → panel 1 only; else listing / outfit / fit card)

   Tools (each owns its failure mode):
     search_listings  → returns []        (pure; no LLM)
     suggest_outfit   → general advice if wardrobe empty; graceful string on API error   (LLM)
     create_fit_card  → error string if outfit empty (no LLM); graceful string on API error (LLM)
```

The error branch is explicit: an empty search result terminates the loop early with
`session["error"]` set and `fit_card`/`outfit_suggestion` left `None`, never reaching
the LLM tools.

---

## AI Tool Plan

I will use ChatGPT lightly as a reviewer, not as the main implementer. The design
decisions live here first: tool inputs and outputs, branch behavior, state keys,
and failure handling.

**Milestone 2: spec check.** I will give ChatGPT the Tools, Planning Loop, State
Management, Error Handling, and Architecture sections and ask whether the
interfaces and branches are specific enough to implement. Expected output: a
short list of missing details or unclear branches. I will only keep feedback that
matches the course requirements.

**Milestone 3: tool edge cases.** For `search_listings`, I will give ChatGPT the
Tool 1 block and ask for edge cases to test. Expected output: a few concrete
inputs, not a full implementation. I will verify them with `pytest tests/`. One
example is making sure size `"S"` does not accidentally match shoe size `"US 7"`.

**Milestone 4: loop sanity check.** I will give ChatGPT the Planning Loop and
State Management sections and ask what should be true after a happy-path run and
after a no-results run. Expected output: a checklist. I will verify it by running
`python agent.py` and checking that the no-results path sets `session["error"]`,
leaves `fit_card` as `None`, and never calls `suggest_outfit`.

Final verification still comes from running the project: `pytest tests/`,
`python agent.py`, and `python app.py`.

---

## A Complete Interaction (Step by Step)

**What FitFindr does (2–3 sentences):** FitFindr is a multi-tool agent that takes a
natural-language thrifting request, *searches* the mock secondhand listings for the
best matches (filtering on keywords, size, and price), then — only if it finds
something — *suggests* a complete outfit pairing the find with the user's existing
wardrobe and *writes* a shareable fit-card caption for it. `search_listings` is
triggered first by every query; `suggest_outfit` and `create_fit_card` fire only
when a listing is found. If the search returns nothing, the agent tells the user
what to change and stops without inventing an outfit.

**Example user query:** "I'm looking for a vintage graphic tee under $30. I mostly wear baggy jeans and chunky sneakers. What's out there and how would I style it?"

**Step 1 — parse:** `run_agent` parses the query → `description="vintage graphic tee"`,
`size=None`, `max_price=30.0`; stores it in `session["parsed"]`.

**Step 2 — search:** calls `search_listings("vintage graphic tee", size=None, max_price=30.0)`.
Several tees score > 0 (e.g. `lst_006` "Graphic Tee — 2003 Tour Bootleg Style" $24
Depop, `lst_033` "Vintage Band Tee — Faded Grey" $19 Depop, `lst_002` "Y2K Baby Tee"
$18 Depop). The highest-scoring item becomes `session["selected_item"]`
(the literal "Graphic Tee" title matches both `graphic` and `tee` plus the `vintage`
tag, so `lst_006` ranks top). Results are non-empty → proceed.

**Step 3 — suggest outfit:** calls `suggest_outfit(selected_item=<that tee>, wardrobe=<example wardrobe>)`.
The LLM returns something like: "Pair this with your baggy straight-leg jeans and
chunky white sneakers for an easy 90s streetwear look; layer the vintage black denim
jacket over it when it's cooler." Stored in `session["outfit_suggestion"]`.

**Step 4 — fit card:** calls `create_fit_card(outfit=<that suggestion>, new_item=<that tee>)`.
The LLM returns something like: "found this faded graphic tee on depop for $24 and
it's already my new everyday — styled it with baggy jeans + chunky sneakers ✨".
Stored in `session["fit_card"]`.

**Final output to user:** the three UI panels populate — **Top listing found**
(title, price, condition, platform, size, why it matched), **Outfit idea** (the
suggestion), and **Your fit card** (the caption). For the no-results query
("designer ballgown size XXS under $5") only the first panel populates, with the
`session["error"]` guidance and no outfit/fit-card.
