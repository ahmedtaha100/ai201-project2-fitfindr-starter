# FitFindr

FitFindr helps a user thrift a piece and style it in one pass. The user types a
request like "vintage graphic tee under $30", the app searches the mock listings,
picks the best match, suggests an outfit using the user's wardrobe, and writes a
short fit-card caption.

The loop is conditional. If the search finds nothing, FitFindr explains what to
change and stops there instead of inventing an outfit from empty input.

- LLM: Groq `llama-3.3-70b-versatile`
- UI: Gradio
- Data: 40 mock listings in `data/listings.json`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY=your_key_here to .env
```

Run the app and checks:

```bash
python app.py
python agent.py
pytest tests/
```

`python app.py` prints the local Gradio URL, usually `http://localhost:7860`.

## Tool inventory

The documented signatures match `tools.py`.

### `search_listings(description, size, max_price) -> list[dict]`

Pure Python search over `data/listings.json`.

Inputs:

- `description` (`str`): words that describe the item, for example
  `"vintage graphic tee"`.
- `size` (`str | None`): optional size filter, for example `"M"`, `"8"`,
  or `"W30"`.
- `max_price` (`float | None`): optional inclusive price ceiling.

Output:

- `list[dict]`: matching listing dictionaries sorted by relevance.
- Each result keeps the dataset fields: `id`, `title`, `description`,
  `category`, `style_tags`, `size`, `condition`, `price`, `colors`, `brand`,
  and `platform`.
- Returns `[]` when nothing matches.

Purpose:

- Filter by price and size.
- Rank the remaining listings by weighted keyword overlap, with the strongest
  weight on `style_tags` and `title`.

### `suggest_outfit(new_item, wardrobe) -> str`

LLM-backed outfit suggestion.

Inputs:

- `new_item` (`dict`): one listing returned by `search_listings`.
- `wardrobe` (`dict`): a dictionary with an `items` list. The list may be empty.

Output:

- `str`: a non-empty outfit suggestion.
- If the wardrobe has items, the suggestion names pieces the user owns.
- If the wardrobe is empty, it gives general styling advice instead.

Purpose:

- Turn the selected listing into a wearable outfit idea.

### `create_fit_card(outfit, new_item) -> str`

LLM-backed caption writer.

Inputs:

- `outfit` (`str`): the suggestion returned by `suggest_outfit`.
- `new_item` (`dict`): the selected listing, used for the title, price, and
  platform.

Output:

- `str`: a short casual caption that mentions the item, price, and platform.
- If `outfit` is blank, it returns a descriptive error string and does not call
  the LLM.

Purpose:

- Convert the outfit suggestion into a shareable fit-card caption.

### `compare_price(item, listings=None) -> str`

Stretch tool.

Inputs:

- `item` (`dict`): the selected listing.
- `listings` (`list[dict] | None`): optional comparison pool. Defaults to the
  full dataset.

Output:

- `str`: a short verdict saying whether the price is below, near, or above the
  average for listings in the same category.

Purpose:

- Give the user a quick read on whether the item looks fairly priced.

## Planning loop

`run_agent()` in `agent.py` owns the flow.

1. Parse the user query into `description`, `size`, and `max_price`.
2. Call `search_listings(description, size, max_price)`.
3. If the results are empty:
   - If a size filter was used, retry once without the size filter.
   - If the retry still finds nothing, set `session["error"]` and return early.
   - `suggest_outfit` and `create_fit_card` are not called on empty results.
4. If results exist, store `session["selected_item"] = results[0]`.
5. Call `suggest_outfit(selected_item, wardrobe)` and store the returned text.
6. Call `create_fit_card(outfit_suggestion, selected_item)` and store the caption.
7. Return the session for the UI.

That branch is the important part. A matchable query runs all three required
tools. An impossible query stops after search with a useful message.

## State management

Each request uses one `session` dictionary. Later steps read the values written
by earlier steps instead of asking the user to re-enter anything.

| Key | Written by | Read by |
| --- | --- | --- |
| `query` | `_new_session` | parser |
| `parsed` | parser | `search_listings` |
| `search_results` | `search_listings` | branch logic |
| `selected_item` | selection step | `suggest_outfit`, `create_fit_card` |
| `wardrobe` | `_new_session` | `suggest_outfit` |
| `outfit_suggestion` | `suggest_outfit` | `create_fit_card` |
| `fit_card` | `create_fit_card` | UI |
| `retry_note` | fallback retry | UI |
| `error` | empty-results branch | UI |

The selected item is the exact dictionary from `search_results[0]`. The tests
check that object identity so the app cannot fake the state handoff.

## Error handling

Each tool handles its own common failure case.

| Tool | Failure case | Behavior |
| --- | --- | --- |
| `search_listings` | No matching listings | Returns `[]`. The loop either retries without size or returns an actionable error. |
| `suggest_outfit` | Empty wardrobe | Returns general styling advice. |
| `create_fit_card` | Blank outfit text | Returns a descriptive error string before any LLM call. |
| `suggest_outfit` / `create_fit_card` | LLM error or empty model response | Returns a non-empty fallback string. |

Triggered examples:

```python
search_listings("designer ballgown", size="XXS", max_price=5)
# []

create_fit_card("", item)
# "Can't write a fit card without an outfit suggestion yet ..."
```

The test suite also covers malformed inputs such as a missing wardrobe `items`
key, a non-string outfit, odd search argument types, and a `None` query.

## Spec reflection

Writing the flow down before coding helped because the control flow was already
decided: search first, branch on results, then pass the selected item through the
rest of the session. That made `run_agent()` straightforward to implement and
easy to test.

The implementation differs from the example in one practical way. The course
walkthrough names a "Faded Band Tee" result, but that exact title is not in the
provided dataset. FitFindr ranks the actual catalog instead. For "vintage graphic
tee", the top match is `lst_006`, "Graphic Tee 2003 Tour Bootleg Style".

I also added a parser that keeps wardrobe context out of the search text. In a
query like "I want a vintage graphic tee under $30. I mostly wear baggy jeans",
the search should focus on the tee, not the jeans.

## AI usage

I used ChatGPT lightly, mostly for checks after I had already made the design
decisions.

One example: I asked for edge cases around the `search_listings` size filter.
That led me to test that `"S"` does not accidentally match `"US 7"`, and I kept
the token-based size match instead of a naive substring check.

Another example: I used it to compare my README outline against the required
sections. I rewrote the wording myself and kept the final README focused on the
actual behavior of the app.

## Tests

```bash
pytest tests/
```

Current result:

```text
33 passed
```
