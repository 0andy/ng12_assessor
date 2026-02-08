"""
Part 2: Conversational Chat Prompts

System and user prompts for Part 2: Conversational Chat.
Includes query rewriting, qualification, refusal, and citation templates.
"""

import re


# ---------------------------------------------------------------------------
# System-level prompt injected at the start of every chat conversation
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """You are a clinical guidelines assistant specialising in \
the NICE NG12 guideline: Suspected Cancer - Recognition and Referral.

You help clinicians and healthcare professionals understand the NG12 recommendations \
through conversational Q&A.

STRICT GROUNDING RULES:
1. ONLY answer based on the NG12 guideline passages provided below.
   Do NOT use your general medical knowledge.
2. Every factual claim must cite a specific source using [Source N] format.
3. Be precise with age thresholds, action types, and clinical criteria.
   Do not paraphrase in a way that changes the clinical meaning.
4. If multiple sources are relevant, cite all of them.
5. Distinguish clearly between:
   - "Suspected cancer pathway referral" (urgent, 2-week wait)
   - "Urgent investigation" (e.g., chest X-ray within 2 weeks)
   - "Consider" recommendations (lower certainty)
6. When quoting criteria, include ALL conditions (age AND symptom AND duration etc.)
   Do not omit qualifying conditions.

CONVERSATION RULES:
7. Use the conversation history for context, but always ground answers
   in the provided guideline passages.
8. If a follow-up question refers to something from a previous turn,
   use the context to understand what is being asked, but still cite
   the guideline passages for your answer.
9. Keep answers focused and clinical. Use clear, professional language.
10. Structure longer answers with clear paragraphs, not bullet points.

MISSING INFORMATION HANDLING:
11. If asked about specific criteria (age thresholds, symptom duration, referral timing) \
that are NOT present in the provided passages:
    - State clearly: "The specific [criteria type] is not found in these passages."
    - Do NOT guess or infer numbers/thresholds.
    - Suggest rephrasing: "Try asking about [specific cancer type] or [specific symptom]."
12. Never fabricate numbers. If a passage says "persistent" without defining duration, \
say "persistent (duration not specified)" rather than inventing "2 weeks".
13. If you cannot fully answer with the evidence provided, acknowledge what you DO know \
from the passages, then clearly state what information is missing."""


# ---------------------------------------------------------------------------
# Canned refusal when retrieved passages are not relevant enough
# ---------------------------------------------------------------------------
CHAT_REFUSE_RESPONSE = """I don't have sufficient evidence in the NG12 guidelines \
to answer this question. The retrieved passages don't appear to be relevant enough \
to provide a grounded response.

Could you try:
- Asking about a specific cancer type (e.g., lung, breast, colorectal)
- Asking about a specific symptom (e.g., haemoptysis, dysphagia, haematuria)
- Asking about referral criteria for a particular age group or risk factor"""


# ---------------------------------------------------------------------------
# Template for partial / low-confidence answers
# ---------------------------------------------------------------------------
CHAT_QUALIFY_TEMPLATE = """Based on the limited evidence found in the NG12 guidelines, \
I can share the following, but please note this may not fully address your question:

{partial_answer}

For a more complete answer, you may want to ask about a specific cancer type, \
symptom, or referral pathway."""


# ---------------------------------------------------------------------------
# Response when the question is outside NG12 scope entirely
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Response for smalltalk messages (greetings, thanks, bye, etc.)
# ---------------------------------------------------------------------------
SMALLTALK_RESPONSE = """Hello! I'm a clinical guidelines assistant \
specialising in the NICE NG12 guideline: Suspected Cancer — Recognition and Referral.

I can help you understand referral criteria, age thresholds, urgent \
investigation pathways, and safety-netting recommendations across all \
cancer types covered by NG12.

Here are a few things you could ask me:
- "Does unexplained haemoptysis require an urgent referral?"
- "What are the criteria for a 2-week-wait referral for breast cancer?"
- "What safety-netting advice does NG12 recommend?"

How can I help you today?"""


# ---------------------------------------------------------------------------
# Response for meta questions about the assistant itself
# ---------------------------------------------------------------------------
META_RESPONSE = """I'm a clinical guidelines assistant specialising in the \
NICE NG12 guideline: Suspected Cancer — Recognition and Referral.

I can answer questions about:
- Which symptoms and risk factors trigger urgent referral for specific cancers
- Age thresholds, investigation pathways, and referral timeframes
- Safety-netting recommendations from the guideline

Important: I am **not** a doctor and I cannot provide a diagnosis or \
treatment advice. My answers are based solely on the published NG12 \
guideline content.

What would you like to know?"""


CHAT_NON_MEDICAL_REDIRECT_RESPONSE = """That's outside what I can help with — \
I'm designed to answer questions about the NICE NG12 guideline for suspected \
cancer recognition and referral.

Try asking something like: "What symptoms warrant an urgent referral for \
lung cancer?"."""


CHAT_OUT_OF_SCOPE_RESPONSE = """This question appears to fall outside the scope of \
the NG12 Suspected Cancer: Recognition and Referral guideline. NG12 covers criteria \
for referring patients with suspected cancer symptoms for urgent investigation or \
specialist assessment.

I can help with questions about:
- Which symptoms trigger urgent referral for specific cancer types
- Age thresholds and risk factors for referral criteria
- The difference between urgent referral and urgent investigation
- Safety netting recommendations"""


# ---------------------------------------------------------------------------
# Response for safety-critical queries (ER, self-diagnosis, self-treatment)
# ---------------------------------------------------------------------------
CHAT_SAFETY_RESPONSE = """I understand your concern, but I'm not able to \
provide emergency medical advice, confirm or rule out a cancer diagnosis, \
or advise you to skip professional medical care.

**If you are experiencing severe, sudden, or worsening symptoms, please \
contact emergency services (999/911) or go to your nearest A&E immediately.**

What I *can* help with is explaining the NG12 guideline criteria for \
referral and investigation. To do that, I'd need:
- Your age and sex
- Specific symptoms you're experiencing
- How long the symptoms have lasted

Would you like to ask about a specific symptom or referral pathway?"""


# ---------------------------------------------------------------------------
# Response for vague / underspecified queries that need more detail
# ---------------------------------------------------------------------------
CHAT_CLARIFY_RESPONSE = """To help you find the right information in the \
NG12 guideline, I need a bit more detail. Could you tell me:

1. **Age** — many referral thresholds are age-specific
2. **Sex** — some criteria differ by sex
3. **Key symptoms** — e.g. unexplained bleeding, persistent cough, \
lump, weight loss, difficulty swallowing, blood in urine
4. **Duration** — how long have the symptoms been present?
5. **Smoking history** — relevant for lung cancer referral criteria
6. **Any red-flag signs?** — unexplained weight loss, persistent \
bleeding, night sweats, new lumps

The more specific you can be, the better I can match your question to \
the NG12 referral and investigation criteria."""


# ---------------------------------------------------------------------------
# Prompt that asks the LLM to rewrite a follow-up message into a standalone
# search query (used by QueryBuilder tier B)
# ---------------------------------------------------------------------------
REWRITE_PROMPT = """Rewrite this message into a standalone search query for NICE NG12 guidelines.

RULES:
1. Do NOT add facts (ages, durations, symptoms) not in the conversation
2. Keep the user's exact medical terms (e.g., "haemoptysis" not "coughing blood")
3. If information is missing, keep the query general - do not guess
4. Under 20 words
5. Do NOT answer - only rewrite for search

Conversation:
{history}

Message: {message}

Query:"""


# ---------------------------------------------------------------------------
# The main user-turn template sent alongside the system prompt
# ---------------------------------------------------------------------------
CHAT_USER_TEMPLATE = """NG12 Guideline Passages:

{context}

---

Conversation History:
{history}

---

Current Question: {message}

---

Instructions:
- Answer using ONLY the guideline passages above
- Cite using [Source N] format for EVERY factual claim
- If the passages don't contain enough evidence, say so explicitly
- Be precise with clinical criteria (age, symptoms, action types)"""


# ===== Helper / formatting functions ========================================


def _get_section(chunk: dict) -> str | None:
    """Get section number from chunk metadata, falling back to canonical.

    Checks 'section' in the chunk's own metadata first, then in
    canonical_metadata (attached at query time by _attach_canonicals).
    Returns None if no section is available (e.g. symptom_index chunks).
    """
    meta = chunk.get("metadata", {})
    section = meta.get("section")
    if not section:
        section = chunk.get("canonical_metadata", {}).get("section")
    return section


def _get_page(chunk: dict) -> int | str:
    """Get page number from chunk metadata, falling back to canonical."""
    meta = chunk.get("metadata", {})
    page = meta.get("page")
    if not page:
        page = chunk.get("canonical_metadata", {}).get("page")
    return page if page else "?"


def _format_citation_ref(chunk: dict) -> str:
    """Build a human-readable citation reference for a chunk.

    Returns:
        'NG12 §1.1.1, p.9'  for rule_search / rule_canonical chunks
        'NG12 Part B, p.43'  for symptom_index chunks
    """
    doc_type = chunk.get("metadata", {}).get("doc_type", "")
    page = _get_page(chunk)

    if doc_type == "symptom_index":
        return f"NG12 Part B, p.{page}"

    section = _get_section(chunk)
    if section:
        return f"NG12 \u00a7{section}, p.{page}"
    return f"NG12 p.{page}"


def format_chat_context(chunks: list[dict]) -> str:
    """Format RAG chunks into numbered context blocks for the prompt.

    Each chunk is rendered with a header containing source index, section,
    page, cancer type, and action type metadata.
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        section = _get_section(chunk) or "Part B"
        page = _get_page(chunk)
        header = (
            f"[Source {i}"
            f" | Section {section}"
            f" | Page {page}"
            f" | {meta.get('cancer_type', 'N/A')}"
            f" | {meta.get('action_type', 'N/A')}]"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def format_history(history: list[dict], max_turns: int = 6) -> str:
    """Format conversation history, keeping only the most recent *max_turns* messages.

    Assistant messages are truncated to 200 characters to save prompt space.
    """
    recent = history[-max_turns:] if len(history) > max_turns else history
    if not recent:
        return "(No previous conversation)"

    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = msg["content"]
        if len(content) > 200 and msg["role"] == "assistant":
            content = content[:200] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_chat_prompt(
    message: str,
    chunks: list[dict],
    history: list[dict],
) -> str:
    """Assemble the full chat user prompt from message, chunks, and history."""
    return CHAT_USER_TEMPLATE.format(
        context=format_chat_context(chunks),
        history=format_history(history),
        message=message,
    )


def build_citations_from_chunks(
    chunks: list[dict],
    answer: str,
) -> list[dict]:
    """Extract [Source N] references from *answer* and map them back to chunk metadata.

    Handles both single-source (``[Source 1]``) and multi-source
    (``[Source 1, 2, 3]``) citation formats.

    Returns only the chunks actually cited.  If the answer contains no
    ``[Source ...]`` markers, returns an empty list (no fake citations).
    """
    cited_indices: set[int] = set()

    # Match single [Source N]
    for match in re.finditer(r"\[Source\s*(\d+)\]", answer):
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(chunks):
            cited_indices.add(idx)

    # Match multi [Source 1, 2, 3, 4]
    for match in re.finditer(r"\[Source\s*([\d,\s]+)\]", answer):
        for num_str in match.group(1).split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                idx = int(num_str) - 1
                if 0 <= idx < len(chunks):
                    cited_indices.add(idx)

    # Do not fake citations when the LLM fails to cite properly
    if not cited_indices:
        return []

    citations = []
    for i in sorted(cited_indices):
        chunk = chunks[i]
        meta = chunk["metadata"]
        section = _get_section(chunk) or "Part B"
        page = _get_page(chunk)
        if page == "?":
            page = 0
        citations.append(
            {
                "source": "NG12 PDF",
                "section": section,
                "page": page,
                "chunk_id": meta.get("chunk_id", "unknown"),
                "excerpt": chunk["text"][:200],
            }
        )
    return citations


def clean_answer_sources(answer: str, chunks: list[dict]) -> str:
    """Replace ``[Source N]`` and ``[Source N, N, ...]`` with readable refs.

    Uses ``_format_citation_ref`` so that rule_search chunks render as
    ``[NG12 §1.1.1, p.9]`` and symptom_index chunks as ``[NG12 Part B, p.43]``.
    """

    def _replace_multi(match: re.Match) -> str:
        """Handle comma-separated indices like [Source 1, 2, 3]."""
        refs: list[str] = []
        for num_str in match.group(1).split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                idx = int(num_str) - 1
                if 0 <= idx < len(chunks):
                    refs.append(_format_citation_ref(chunks[idx]))
        if refs:
            return "[" + "; ".join(refs) + "]"
        return match.group(0)

    def _replace_single(match: re.Match) -> str:
        """Handle a single index like [Source 1]."""
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(chunks):
            return f"[{_format_citation_ref(chunks[idx])}]"
        return match.group(0)

    # First pass: multi-source references  [Source 1, 2, 3]
    answer = re.sub(r"\[Source\s*([\d,\s]+)\]", _replace_multi, answer)
    # Second pass: any remaining single-source references  [Source 1]
    answer = re.sub(r"\[Source\s*(\d+)\]", _replace_single, answer)
    return answer
