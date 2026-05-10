"""Lightweight keyword extraction for chunk enrichment."""

import re
import string
from collections import Counter


# Common English stop words — avoids heavy NLP dependency
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "this", "that", "these", "those", "it", "its",
    "as", "if", "than", "then", "so", "not", "no", "nor", "yet", "both",
    "either", "each", "every", "all", "any", "more", "most", "also", "just",
    "about", "which", "who", "what", "when", "where", "how", "their", "they",
    "we", "our", "you", "your", "he", "she", "his", "her", "i", "my", "me",
}


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Extract the top_n most significant terms from text using TF-based scoring.

    Keeps multi-word capitalized phrases (likely proper nouns / named entities)
    and high-frequency non-stop single tokens.
    """
    # Normalize
    text_lower = text.lower()
    tokens = re.findall(r'\b[a-z][a-z0-9\-]{2,}\b', text_lower)

    # Remove stop words and very short tokens
    filtered = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

    # Frequency ranking
    freq = Counter(filtered)
    top_single = [word for word, _ in freq.most_common(top_n)]

    # Also capture capitalized multi-word phrases (e.g. "APAC Region", "Q3 Revenue")
    phrases = re.findall(r'\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]+)+\b', text)
    phrases = list(dict.fromkeys(phrases))[:5]  # deduplicate, cap at 5

    # Merge: phrases first, then single-word terms
    seen: set[str] = set()
    keywords: list[str] = []
    for kw in phrases + top_single:
        norm = kw.lower()
        if norm not in seen:
            seen.add(norm)
            keywords.append(kw)
        if len(keywords) >= top_n:
            break

    return keywords


def enrich_chunk_for_embedding(text: str, keywords: list[str]) -> str:
    """
    Prepend extracted keywords to chunk text before embedding.
    The enriched text is only used for the embedding step — NOT stored in ChromaDB.
    """
    if not keywords:
        return text
    keyword_line = "Keywords: " + ", ".join(keywords)
    return f"{keyword_line}\n\n{text}"
