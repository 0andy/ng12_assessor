"""
Session Store

Manages conversation history and topic state for Part 2 chat.
Provides a SessionStore class with methods for history, topic tracking,
and session cleanup.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


# Clinical terms used for topic extraction from chunk text.
# Covers symptoms, investigation types, and cancer-type names so the topic
# string is maximally useful when prepended to a follow-up query.
_CLINICAL_TERMS: list[str] = [
    # Symptoms
    "haemoptysis", "hemoptysis", "dysphagia", "haematuria", "hematuria",
    "lymphadenopathy", "hoarseness", "breast lump", "weight loss",
    "chest x-ray", "referral", "investigation", "endoscopy",
    "ultrasound", "anaemia", "jaundice",
    # Cancer-type keywords (useful for topic enrichment)
    "lung", "breast", "colorectal", "prostate", "skin", "melanoma",
    "sarcoma", "leukaemia", "lymphoma", "myeloma", "pancreatic",
    "ovarian", "bladder", "renal", "testicular", "thyroid", "brain",
]

# cancer_type metadata values that are NOT actual cancer types.
# These appear on support/preamble/general NG12 sections and should
# be excluded when building the topic string for query enrichment.
_NON_CANCER_TYPES: set[str] = {
    "General",
    "Patient information and support",
    "Safety netting",
    "Overview",
    "Introduction",
    "N/A",
    "",
}


class SessionStore:
    """In-memory store for chat session history and topic state."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._topics: dict[str, str] = {}

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve conversation history for a session.

        Args:
            session_id: The unique session identifier.

        Returns:
            List of message dicts with keys: role, content.
        """
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history.

        Args:
            session_id: The unique session identifier.
            role: Either "user" or "assistant".
            content: The message text.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})

    def get_topic(self, session_id: str) -> str:
        """Get the current topic for a session.

        Args:
            session_id: The unique session identifier.

        Returns:
            The current topic string, or empty string if not set.
        """
        return self._topics.get(session_id, "")

    def update_topic(self, session_id: str, chunks: list[dict[str, Any]]) -> None:
        """Update the current topic from retrieved chunks.

        Builds a space-separated topic string that is useful when prepended
        to a follow-up query.  Only non-general chunks are used so that
        generic preamble / support sections do not pollute the topic.

        Extracts:
          - Most common cancer_type across relevant chunks
          - Up to 3 unique section numbers
          - Up to 3 clinical terms found in chunk text

        Args:
            session_id: The unique session identifier.
            chunks: List of retrieval result dicts with metadata.
        """
        if not chunks:
            return

        # Filter to cancer-specific chunks only (exclude general / support)
        relevant = [
            c for c in chunks
            if c["metadata"].get("cancer_type", "General") not in _NON_CANCER_TYPES
            and c["metadata"].get("section", "general") != "general"
        ]
        if not relevant:
            # Fallback: use all chunks but still try to extract useful info
            relevant = chunks

        # --- Most common cancer_type ---
        cancer_types = [
            c["metadata"].get("cancer_type", "")
            for c in relevant
            if c["metadata"].get("cancer_type", "") not in _NON_CANCER_TYPES
        ]
        top_cancer = ""
        if cancer_types:
            top_cancer = Counter(cancer_types).most_common(1)[0][0]

        # --- Unique section numbers (up to 3, prefer cancer sections 1.1-1.13) ---
        sections: list[str] = []
        seen: set[str] = set()
        for c in relevant:
            sec = c["metadata"].get("section", "")
            if sec and sec != "general" and sec not in seen:
                seen.add(sec)
                sections.append(sec)
            if len(sections) >= 3:
                break

        # --- Clinical terms from chunk text (up to 3) ---
        keywords: list[str] = []
        for c in relevant:
            text_lower = c.get("text", "").lower()
            for term in _CLINICAL_TERMS:
                if term in text_lower and term not in keywords:
                    keywords.append(term)
                    if len(keywords) >= 3:
                        break
            if len(keywords) >= 3:
                break

        # Build a space-separated topic string (good for query prepending).
        # Section numbers (e.g. "1.1.1") are stored internally but NOT
        # included in the topic string because they add noise to vector
        # search queries.  Only the cancer type and clinical keywords are
        # useful for semantic retrieval enrichment.
        parts: list[str] = []
        if top_cancer:
            parts.append(top_cancer)
        parts.extend(keywords[:2])

        topic = " ".join(parts) if parts else "general"
        self._topics[session_id] = topic

        print(f"[SessionStore] Updated topic for {session_id}: '{topic}'")

    def clear_session(self, session_id: str) -> None:
        """Clear a single session's history and topic.

        Args:
            session_id: The unique session identifier.
        """
        self._sessions.pop(session_id, None)
        self._topics.pop(session_id, None)

    def clear_all(self) -> None:
        """Clear all sessions and topics."""
        self._sessions.clear()
        self._topics.clear()


# Module-level singleton instance
session_store = SessionStore()
