"""
Tiered Query Builder for Part 2 Chat

Implements an A + C + B strategy for building retrieval queries:
  A (direct)         - Use the raw user message as-is.
  C (topic_enriched) - Prepend the session topic keyword to a follow-up message.
  B (llm_rewrite)    - Fall back to an LLM rewrite when no session topic is available.
"""

from __future__ import annotations

import re

from app.prompts.chat import REWRITE_PROMPT, format_history


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

# Phrases that typically start a follow-up / context-dependent message
_FOLLOWUP_STARTERS: list[str] = [
    "what about",
    "and if",
    "how about",
    "what if",
    "and for",
    "but what",
    "also",
    "same for",
    "does that",
    "is that",
    "can you",
    "could you",
    "what's the",
    "how does",
    "earlier",
    "you mentioned",
    "you said",
    "go back",
    "going back",
]

# Pronouns that signal the message depends on earlier context
_CONTEXT_PRONOUNS: set[str] = {
    "it", "that", "they", "this", "them", "those", "same",
}

# Regex to strip punctuation (keeps letters, digits, spaces, and hyphens)
_STRIP_PUNCT_RE = re.compile(r"[^\w\s\-]", re.UNICODE)


def is_followup(message: str) -> bool:
    """Detect whether *message* is a short follow-up that needs context enrichment.

    A message is considered a follow-up if ANY of the following is true:
    1. It contains 3 words or fewer (after stripping punctuation).
    2. It starts with a known follow-up phrase.
    3. It is shorter than 12 words AND contains a context-dependent pronoun.
    """
    msg_lower = message.lower().strip()

    # Strip punctuation before splitting into words so that trailing "?"
    # does not inflate the word count (e.g. "smokers?" -> "smokers").
    msg_cleaned = _STRIP_PUNCT_RE.sub("", msg_lower)
    words = msg_cleaned.split()

    # Condition 1: very short utterance
    very_short = len(words) <= 3

    # Condition 2: starts with a follow-up phrase
    starts_with_followup = any(msg_lower.startswith(s) for s in _FOLLOWUP_STARTERS)

    # Condition 3: sentence with a context pronoun (under 12 words)
    short = len(words) < 12
    has_pronoun = bool(_CONTEXT_PRONOUNS.intersection(words))

    result = very_short or starts_with_followup or (short and has_pronoun)

    print(
        f"[QueryBuilder] is_followup check: '{message}', "
        f"words={len(words)}, very_short={very_short}, "
        f"starts_followup={starts_with_followup}, "
        f"has_pronoun={has_pronoun}, result={result}"
    )

    return result


# ---------------------------------------------------------------------------
# QueryBuilder
# ---------------------------------------------------------------------------

class QueryBuilder:
    """Build the retrieval query for a chat turn using the A+C+B strategy.

    Parameters
    ----------
    session_store:
        Object that exposes ``get_topic(session_id)`` and
        ``get_history(session_id)`` methods.
    gemini_client:
        Optional LLM client with ``is_available`` (bool) and
        ``async generate(system_prompt, user_prompt)`` interface.
    """

    def __init__(self, session_store, gemini_client=None):
        self.session_store = session_store
        self.gemini = gemini_client

    async def build(
        self,
        session_id: str,
        message: str,
    ) -> tuple[str, str]:
        """Build a retrieval query for *message* within *session_id*.

        Returns
        -------
        tuple[str, str]
            ``(query, strategy)`` where *strategy* is one of
            ``"direct"`` / ``"topic_enriched"`` / ``"llm_rewrite"``.
        """

        # --- Tier A: standalone message -> use as-is ---
        followup = is_followup(message)
        topic = self.session_store.get_topic(session_id)

        if not followup:
            # Short messages (≤5 words) with an active topic are likely
            # context-dependent (e.g. bare symptom additions like
            # "headache for two days") even without explicit follow-up
            # markers.  Enrich them to maintain conversational context.
            msg_cleaned = _STRIP_PUNCT_RE.sub("", message.lower().strip())
            if topic and len(msg_cleaned.split()) <= 5:
                enriched = f"{topic} {message}"
                print(f"[QueryBuilder] Short-with-topic enrichment: '{enriched}'")
                return enriched, "topic_enriched"
            return message, "direct"

        # --- Tier C: follow-up + known topic -> prepend topic keyword ---
        if topic:
            enriched = f"{topic} {message}"
            return enriched, "topic_enriched"

        # --- Tier B: fallback -> LLM-based rewrite ---
        if self.gemini and self.gemini.is_available:
            history = self.session_store.get_history(session_id)
            if history:
                history_text = format_history(history, max_turns=6)
                rewrite_prompt = REWRITE_PROMPT.format(
                    history=history_text,
                    message=message,
                )
                try:
                    rewritten = await self.gemini.generate(
                        system_prompt="",
                        user_prompt=rewrite_prompt,
                    )
                    if rewritten and rewritten.strip():
                        return rewritten.strip(), "llm_rewrite"
                except Exception as exc:  # noqa: BLE001
                    print(f"LLM rewrite failed: {exc}")

        # --- Ultimate fallback: return the raw message ---
        return message, "direct"
