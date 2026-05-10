"""General-purpose utility functions."""

import hashlib
import re


def slugify(text: str) -> str:
    """Convert arbitrary text to a URL-safe slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)


def stable_id(*parts: str) -> str:
    """Deterministic SHA-256 ID from one or more string parts (e.g. file_id + chunk_index)."""
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:32]


def truncate(text: str, max_chars: int = 200, suffix: str = "…") -> str:
    """Truncate text to max_chars without cutting mid-word."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + suffix


def chunk_list(lst: list, size: int) -> list[list]:
    """Split a list into consecutive chunks of at most `size` items."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]
