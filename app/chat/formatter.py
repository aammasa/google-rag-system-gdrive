"""
Google Chat response formatter.

Builds cardsV2 payloads so answers are rendered with rich formatting —
header, answer text, and clickable source buttons — instead of plain text.

Reference: https://developers.google.com/chat/api/guides/message-formats/cards
"""

import textwrap


def build_answer_card(answer: str, sources: list[dict]) -> dict:
    """
    Build a cardsV2 response with the RAG answer and source buttons.

    Args:
        answer:  The generated answer text.
        sources: List of source dicts with keys: file_name, web_view_link, score.
    """
    sections = []

    # ── Answer section ────────────────────────────────────────────────────────
    # Google Chat renders basic HTML: <b>, <i>, <br>, <a href>
    formatted_answer = answer.replace("\n", "<br>")
    sections.append({
        "widgets": [
            {"textParagraph": {"text": formatted_answer}}
        ]
    })

    # ── Sources section ───────────────────────────────────────────────────────
    if sources:
        source_buttons = []
        for src in sources:
            name = src.get("file_name") or "Source document"
            link = src.get("web_view_link", "")
            score = src.get("score", 0)
            label = f"{textwrap.shorten(name, width=40)} ({score:.0%})"

            if link:
                source_buttons.append({
                    "text": label,
                    "onClick": {"openLink": {"url": link}},
                })
            else:
                source_buttons.append({"text": label})

        sections.append({
            "header": "Sources",
            "collapsible": True,
            "widgets": [{"buttonList": {"buttons": source_buttons}}],
        })

    return {
        "cardsV2": [{
            "cardId": "rag-response",
            "card": {
                "header": {
                    "title": "Knowledge Base Answer",
                    "subtitle": "Sourced from Google Drive",
                    "imageUrl": "https://fonts.gstatic.com/s/i/googlematerialicons/search/v15/googlematerialicons-search-48dp/1x/gm_search_black_48dp.png",
                    "imageType": "CIRCLE",
                },
                "sections": sections,
            },
        }]
    }


def build_text_response(text: str) -> dict:
    """Plain text response — used for slash command confirmations."""
    return {"text": text}


def build_help_card() -> dict:
    """Build the /help command response card."""
    return {
        "cardsV2": [{
            "cardId": "help-card",
            "card": {
                "header": {"title": "Available Commands"},
                "sections": [{
                    "widgets": [
                        {"textParagraph": {"text": (
                            "<b>/help</b> — Show this help message<br>"
                            "<b>/clear</b> — Clear your conversation history<br>"
                            "<b>/sources</b> — Show sources from the last answer<br>"
                            "<b>/ingest</b> — Trigger a re-ingestion of Drive documents<br>"
                            "<br>"
                            "Or just ask me anything about your Google Drive documents!"
                        )}}
                    ]
                }]
            }
        }]
    }


def build_error_card(message: str) -> dict:
    """Build an error response card."""
    return {
        "cardsV2": [{
            "cardId": "error-card",
            "card": {
                "sections": [{
                    "widgets": [
                        {"textParagraph": {"text": f"⚠️ {message}"}}
                    ]
                }]
            }
        }]
    }
