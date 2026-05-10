"""
In-memory conversation session store.

Stores turn history per session_id with TTL expiry and a max-turns cap.
Replace the backing store with Redis for multi-instance / Cloud Run deployments.
"""

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class Turn:
    role: str       # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Session:
    turns: list[Turn] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionStore:
    """
    Thread-safe in-memory session store.

    Args:
        ttl_seconds:  Sessions inactive longer than this are expired (default 1 hr).
        max_turns:    Maximum number of turns kept per session (oldest dropped first).
    """

    def __init__(self, ttl_seconds: int = 3600, max_turns: int = 20) -> None:
        self._sessions: dict[str, Session] = {}
        self._ttl = ttl_seconds
        self._max_turns = max_turns
        self._lock = Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_history(self, session_id: str) -> list[Turn]:
        """Return the turn list for a session (empty list if not found / expired)."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or self._is_expired(session):
                return []
            return list(session.turns)

    def append(self, session_id: str, role: str, content: str) -> None:
        """Add a turn to the session, creating it if necessary."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or self._is_expired(session):
                session = Session()
                self._sessions[session_id] = session

            session.turns.append(Turn(role=role, content=content))
            session.last_active = time.time()

            # Keep only the most recent max_turns
            if len(session.turns) > self._max_turns:
                session.turns = session.turns[-self._max_turns :]

    def clear(self, session_id: str) -> None:
        """Delete a session entirely."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_ids(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def purge_expired(self) -> int:
        """Remove all expired sessions. Returns count removed."""
        with self._lock:
            expired = [sid for sid, s in self._sessions.items() if self._is_expired(s)]
            for sid in expired:
                del self._sessions[sid]
            return len(expired)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_expired(self, session: Session) -> bool:
        return (time.time() - session.last_active) > self._ttl
