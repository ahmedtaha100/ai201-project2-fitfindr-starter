"""
app.py

Gradio interface for FitFindr. The layout and wiring are already set up —
your job is to fill in handle_query() so it calls run_agent() and maps
the session results to the three output panels.

Run with:
    python app.py

Then open the localhost URL shown in your terminal (usually http://localhost:7860,
but check your terminal — the port may differ).
"""

import gradio as gr

from agent import run_agent
from tools import compare_price
from utils.data_loader import get_example_wardrobe, get_empty_wardrobe


# ── query handler ─────────────────────────────────────────────────────────────

def _format_listing_panel(session: dict) -> str:
    """Render session['selected_item'] (+ stretch price check) as readable text."""
    item = session["selected_item"] or {}
    price = item.get("price")
    price_str = f"${price:g}" if isinstance(price, (int, float)) else "price n/a"
    lines = []
    if session.get("retry_note"):
        lines.append(f"ℹ️ {session['retry_note']}\n")
    brand = f"   |   Brand: {item['brand']}" if item.get("brand") else ""
    lines += [
        item.get("title", "(untitled listing)"),
        f"{price_str}  ·  {item.get('condition', '?')} condition  ·  {item.get('platform', '?')}",
        f"Size: {item.get('size', '?')}   |   Category: {item.get('category', '?')}",
        f"Style: {', '.join(item.get('style_tags', []) or [])}",
        f"Colors: {', '.join(item.get('colors', []) or [])}{brand}",
        "",
        item.get("description", ""),
        "",
        f"💲 {compare_price(item)}",
    ]
    return "\n".join(lines)


def handle_query(user_query: str, wardrobe_choice: str) -> tuple[str, str, str]:
    """
    Called by Gradio when the user submits a query.

    Args:
        user_query:      The text the user typed into the search box.
        wardrobe_choice: Either "Example wardrobe" or "Empty wardrobe (new user)".

    Returns:
        A tuple of three strings (listing_text, outfit_suggestion, fit_card),
        one per output panel. On the no-results path, the error goes in the
        first panel and the other two are empty.
    """
    # 1. Guard an empty query.
    if not user_query or not user_query.strip():
        return (
            "Please type what you're looking for "
            "(e.g. “vintage graphic tee under $30, size M”).",
            "",
            "",
        )

    # 2. Select the wardrobe.
    wardrobe = (
        get_empty_wardrobe()
        if str(wardrobe_choice).lower().startswith("empty")
        else get_example_wardrobe()
    )

    # 3. Run the planning loop.
    session = run_agent(user_query.strip(), wardrobe)

    # 4. Error path -> message in panel 1 only.
    if session["error"]:
        return session["error"], "", ""

    # 5. Happy path -> map the session to the three panels.
    return (
        _format_listing_panel(session),
        session["outfit_suggestion"] or "",
        session["fit_card"] or "",
    )


# ── interface ─────────────────────────────────────────────────────────────────

EXAMPLE_QUERIES = [
    "vintage graphic tee under $30",
    "90s track jacket in size M",
    "flowy midi skirt under $40",
    "black combat boots size 8",
    "designer ballgown size XXS under $5",   # deliberate no-results test
]

def build_interface():
    with gr.Blocks(title="FitFindr") as demo:
        gr.Markdown("""
# FitFindr 🛍️
Find secondhand pieces and get outfit ideas based on your wardrobe.
Describe what you're looking for — include size and price if you want to filter.
        """)

        with gr.Row():
            query_input = gr.Textbox(
                label="What are you looking for?",
                placeholder="e.g. vintage graphic tee under $30, size M",
                lines=2,
                scale=3,
            )
            wardrobe_choice = gr.Radio(
                choices=["Example wardrobe", "Empty wardrobe (new user)"],
                value="Example wardrobe",
                label="Wardrobe",
                scale=1,
            )

        submit_btn = gr.Button("Find it", variant="primary")

        with gr.Row():
            listing_output = gr.Textbox(
                label="🛍️ Top listing found",
                lines=8,
                interactive=False,
            )
            outfit_output = gr.Textbox(
                label="👗 Outfit idea",
                lines=8,
                interactive=False,
            )
            fitcard_output = gr.Textbox(
                label="✨ Your fit card",
                lines=8,
                interactive=False,
            )

        gr.Examples(
            examples=[[q, "Example wardrobe"] for q in EXAMPLE_QUERIES],
            inputs=[query_input, wardrobe_choice],
            label="Try these queries",
        )

        submit_btn.click(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )
        query_input.submit(
            fn=handle_query,
            inputs=[query_input, wardrobe_choice],
            outputs=[listing_output, outfit_output, fitcard_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
