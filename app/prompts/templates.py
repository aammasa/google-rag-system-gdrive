"""Prompt templates for the RAG pipeline."""

from app.core.session_store import Turn

RAG_SYSTEM_PROMPT = """You are a helpful assistant with access to a company's internal \
knowledge base sourced from Google Drive.

Answer the user's question using ONLY the context provided below. \
If the answer is not in the context, say so honestly — do not fabricate information.

When citing information, reference the source document name.
"""

RAG_CONTEXT_TEMPLATE = """
=== Retrieved Context ===
{context}
========================
"""

RAG_USER_TEMPLATE = "Question: {question}"

CONDENSE_QUESTION_PROMPT = """Given the conversation history below and a follow-up question, \
rewrite the follow-up as a fully self-contained standalone question. \
Preserve all relevant details from the history needed to answer it. \
Return ONLY the rewritten question — no explanation, no preamble.

Conversation history:
{chat_history}

Follow-up question: {question}
Standalone question:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_history_for_condensation(turns: list[Turn]) -> str:
    """Render turn history as a plain text block for the condensation prompt."""
    lines = []
    for turn in turns:
        prefix = "User" if turn.role == "user" else "Assistant"
        lines.append(f"{prefix}: {turn.content}")
    return "\n".join(lines)


def history_to_langchain_messages(turns: list[Turn]) -> list:
    """
    Convert stored turns into LangChain message objects for the final LLM call.
    Alternating HumanMessage / AIMessage lets Gemini understand the conversation flow.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    messages = []
    for turn in turns:
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    return messages


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    """Assemble the full RAG prompt (used outside agent for testing)."""
    context = "\n\n---\n\n".join(context_chunks)
    return (
        RAG_SYSTEM_PROMPT
        + RAG_CONTEXT_TEMPLATE.format(context=context)
        + RAG_USER_TEMPLATE.format(question=question)
    )
