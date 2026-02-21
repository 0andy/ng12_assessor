"""
Part 2: Conversational Chat Workflow

LangGraph workflow:
  load_history -> input_guardrail
    -> [smalltalk/meta/chitchat]  -> smalltalk_meta -> save_history
    -> [safety_urgent]            -> smalltalk_meta -> save_history
    -> [needs_clarification]      -> smalltalk_meta -> save_history
    -> [medical_out_of_scope]     -> out_of_scope   -> save_history
    -> [proceed]                  -> build_query -> retrieve -> guardrail_check
        -> [sufficient/weak]     -> summarize_query -> generate / qualify -> save_history
        -> [none]                -> refuse -> save_history
        -> [out_of_scope]        -> out_of_scope -> save_history

Handles multi-turn conversation with topic tracking, tiered query building,
and grounding guardrails that determine response quality tier.
Input guardrail short-circuits smalltalk and meta queries to avoid
unnecessary RAG pipeline processing.
The summarize_query node extracts structured clinical info for display only
(does not affect retrieval).
"""

from __future__ import annotations

import logging
import re
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.core import rag_pipeline
from app.core.gemini_client import gemini_client
from app.core.query_builder import QueryBuilder
from app.memory.session_store import session_store
from app.prompts.chat import (
    CHAT_CLARIFY_RESPONSE,
    CHAT_NON_MEDICAL_REDIRECT_RESPONSE,
    CHAT_OUT_OF_SCOPE_RESPONSE,
    CHAT_QUALIFY_TEMPLATE,
    CHAT_REFUSE_RESPONSE,
    CHAT_SAFETY_RESPONSE,
    CHAT_SYSTEM_PROMPT,
    META_RESPONSE,
    REWRITE_PROMPT,
    SMALLTALK_RESPONSE,
    build_citations_from_chunks,
    clean_answer_sources,
    format_chat_prompt,
    format_history,
)

logger = logging.getLogger(__name__)

# Module-level QueryBuilder singleton
query_builder = QueryBuilder(session_store, gemini_client)

# Keywords indicating the question is about topics NG12 does NOT cover
_OUT_OF_SCOPE_KEYWORDS: list[str] = [
    "treatment", "chemotherapy", "prognosis", "survival rate",
    "medication", "drug", "cure", "surgery", "radiotherapy",
    "immunotherapy", "dosage", "side effect", "stage", "staging",
    "metastasis", "palliative",
]

# Keywords indicating the question IS about NG12 scope (recognition & referral)
_IN_SCOPE_KEYWORDS: list[str] = [
    "referral", "refer", "investigation", "symptom", "recognition",
    "criteria", "threshold", "age", "guideline", "ng12",
    "suspected cancer", "pathway", "urgent", "safety net",
]

# --- Input-level medical out-of-scope keyword groups ---
# Treatment-related queries outside NG12 scope
_TREATMENT_KEYWORDS: list[str] = [
    "chemotherapy", "radiotherapy", "immunotherapy", "surgery",
    "medication", "drug", "cure", "therapy", "treat",
]
# Prognosis-related queries outside NG12 scope
_PROGNOSIS_KEYWORDS: list[str] = [
    "prognosis", "survival rate", "life expectancy", "outcome",
    "mortality", "survive",
]
# Self-diagnosis queries outside NG12 scope
_DIAGNOSIS_PHRASES: list[str] = [
    "do i have cancer", "diagnose me", "is this cancer",
    "could this be cancer",
]
# Referral-context keywords that override out-of-scope classification
_REFERRAL_CONTEXT_KEYWORDS: list[str] = [
    "referral", "refer", "investigation", "criteria", "symptom", "sign",
]

# Common stop-words excluded from the lexical overlap check
_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "in", "on",
    "at", "to", "for", "of", "and", "or", "with", "what", "how",
    "does", "do", "can", "about", "tell", "me", "that", "this",
    "it", "be", "not", "no", "by", "from", "but", "if", "so",
    "my", "you", "your", "i", "we", "they", "he", "she",
}


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ChatState(TypedDict):
    session_id: str
    message: str
    history: Optional[list]
    search_query: Optional[str]
    query_strategy: Optional[str]
    chunks: Optional[list]
    answer: Optional[str]
    citations: Optional[list]
    citation_count: Optional[int]
    score_breakdown: Optional[dict]
    guardrail_result: Optional[str]
    query_summary: Optional[str]


# ---------------------------------------------------------------------------
# Node 1: load_history
# ---------------------------------------------------------------------------

async def load_history_node(state: ChatState) -> dict:
    """Load conversation history for the current session."""
    history = session_store.get_history(state["session_id"])
    return {"history": history}


# ---------------------------------------------------------------------------
# Input classification: smalltalk / meta / proceed
# ---------------------------------------------------------------------------

# Patterns for greeting / farewell / pleasantries
_SMALLTALK_PATTERNS: list[re.Pattern] = [
    re.compile(r"^(hi|hello|hey|howdy|hiya|yo)[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(hi|hello|hey)\s+there[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^good\s+(morning|afternoon|evening|day)[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(thanks|thank\s*you|cheers|ta)[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(bye|goodbye|see\s*you|farewell)[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(ok|okay|sure|great|nice|cool|got\s*it)[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^how\s+are\s+you(\s+doing)?[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^how\s+are\s+you\s+today[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(are\s+)?you\s+there[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^what'?s\s+up[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^sup[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(good|fine|great)\s+thanks[\s!.,?]*$", re.IGNORECASE),
    re.compile(r"^(lol|haha|hehe|😂|🤣)[\s!.,?]*$", re.IGNORECASE),
]

# Patterns for questions about the assistant itself
_META_PATTERNS: list[re.Pattern] = [
    re.compile(r"who\s+are\s+you", re.IGNORECASE),
    re.compile(r"what\s+are\s+you", re.IGNORECASE),
    re.compile(r"what\s+can\s+you\s+do", re.IGNORECASE),
    re.compile(r"how\s+do(es)?\s+(this|you)\s+work", re.IGNORECASE),
    re.compile(r"what\s+is\s+this(\s+tool|\s+system|\s+assistant)?[\s?]*$", re.IGNORECASE),
    re.compile(r"tell\s+me\s+about\s+(yourself|this\s+system)", re.IGNORECASE),
    re.compile(r"what\s+do\s+you\s+know", re.IGNORECASE),
    re.compile(r"^help[\s!?]*$", re.IGNORECASE),
    re.compile(r"are\s+you\s+a\s+doctor", re.IGNORECASE),
    re.compile(r"can\s+you\s+diagnose", re.IGNORECASE),
]

# Patterns for non-medical chitchat (jokes, weather, time, sports, etc.)
_CHITCHAT_PATTERNS: list[re.Pattern] = [
    re.compile(r"tell\s+me\s+a\s+joke", re.IGNORECASE),
    re.compile(r"joke", re.IGNORECASE),
    re.compile(r"weather", re.IGNORECASE),
    re.compile(r"what\s*time\s+is\s+it", re.IGNORECASE),
    re.compile(r"time\s+now", re.IGNORECASE),
    re.compile(r"what('?s| is)\s+the\s+date", re.IGNORECASE),
    re.compile(r"(sports?|football|soccer)\s+score", re.IGNORECASE),
    re.compile(r"(remember|know)\s+my\s+name", re.IGNORECASE),
    re.compile(r"what\s+(kind|type)\s+of\s+(ai|model|llm)", re.IGNORECASE),
    re.compile(r"how\s+(were|are)\s+you\s+(built|made|created|trained)", re.IGNORECASE),
    re.compile(r"explain\s+how\s+you\s+(were|are)\s+(built|made|created)", re.IGNORECASE),
    re.compile(r"can\s+you\s+(explain|tell)\s+how\s+you\s+(were|are)\s+(built|made)", re.IGNORECASE),
]

# Patterns for safety-critical queries (ER, self-diagnosis, self-treatment)
_SAFETY_URGENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bemergency\s+(room|department)\b", re.IGNORECASE),
    re.compile(r"\bgo\s+to\s+(the\s+)?(er|a&e)\b", re.IGNORECASE),
    re.compile(r"\bshould\s+i\s+(go\s+to|visit)\s+(the\s+)?(er|a&e|emergency)\b", re.IGNORECASE),
    re.compile(r"\bcall\s+(911|999|an?\s+ambulance)\b", re.IGNORECASE),
    re.compile(r"\bskip\s+(seeing\s+)?(a\s+)?doctor\b", re.IGNORECASE),
    re.compile(r"\bdefinitely\b.{0,20}\bcancer\b", re.IGNORECASE),
    re.compile(r"\bconfirm\b.{0,30}\bcancer\b", re.IGNORECASE),
    re.compile(r"\bcancer\b.{0,30}\bnot\s+(just\s+)?anxiety\b", re.IGNORECASE),
    re.compile(r"\b(treat|manage)\b.{0,15}\b(myself|at\s+home|on\s+my\s+own)\b", re.IGNORECASE),
    re.compile(r"\bself[\s-]?treat\b", re.IGNORECASE),
]

# Vague symptom patterns that need clarification before RAG
_VAGUE_SYMPTOM_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(feel(ing)?|felt)\s+(unwell|sick|ill|bad|off|wrong|funny|strange)\b", re.IGNORECASE),
    re.compile(r"\bsomething\s+(is\s+|feels?\s+)?(wrong|off)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+feeling\s+(well|right|good|myself|great)\b", re.IGNORECASE),
    re.compile(r"\b(been|feeling|very|so|really|quite)\s+(tired|exhausted|fatigued)\b", re.IGNORECASE),
    re.compile(r"\b(tired|exhausted|fatigued)\s+(lately|recently|all\s+the\s+time)\b", re.IGNORECASE),
    re.compile(r"\bis\s+(that|this|it)\s+cancer\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+should\s+i\s+do\b", re.IGNORECASE),
]

# Specific NG12 symptoms — if present, skip needs_clarification
_SPECIFIC_NG12_SYMPTOMS: list[str] = [
    "haematuria", "hematuria", "dysphagia", "haemoptysis", "hemoptysis",
    "lymphadenopathy", "hoarseness", "jaundice", "anaemia", "anemia",
    "dyspepsia", "night sweats", "rectal bleeding", "breast lump",
    "weight loss", "abdominal mass", "abdominal pain", "chest pain",
    "haematemesis", "mole", "lesion", "ulcer",
    "bruising", "petechiae", "hepatomegaly", "splenomegaly",
    "ascites", "pleural effusion", "bone pain", "lump",
]

# Medical signal words — if present, skip smalltalk/chitchat classification
_MEDICAL_SIGNAL_WORDS: list[str] = [
    "referral", "refer", "urgent", "symptom", "cancer",
    "haemoptysis", "dysphagia", "haematuria", "lump",
    "hoarseness", "mole", "bleeding", "weight loss",
    "investigation", "pathway", "guideline", "ng12",
    "suspected", "criteria", "threshold", "safety net",
    "age", "diagnosis", "rectal", "breast", "lung",
    "colorectal", "prostate", "ovarian", "pancreatic",
    "oesophageal", "bladder", "renal", "melanoma",
]


def _has_medical_signal(text_lower: str) -> bool:
    """Return True if the message contains NG12-relevant medical terms.

    Used to prevent smalltalk / chitchat patterns from capturing messages
    that are actually clinical questions (e.g. "hi, does haemoptysis
    require urgent referral?").
    """
    return any(kw in text_lower for kw in _MEDICAL_SIGNAL_WORDS)


def classify_input(message: str) -> str:
    """Classify a user message before it enters the RAG pipeline.

    Uses deterministic keyword/regex matching only — no LLM calls.

    Priority order:
        1. smalltalk (greetings, thanks, farewells) — only if no medical signal
        2. meta (questions about the assistant)
        3. chitchat_redirect (jokes, weather, time, sports, etc.)
        4. safety_urgent (ER, self-diagnosis confirmation, self-treatment)
        5. medical_out_of_scope (treatment, prognosis, self-diagnosis)
        6. needs_clarification (vague symptoms, no specifics)
        7. proceed (everything else → RAG pipeline)

    Returns one of:
        "smalltalk", "meta", "chitchat_redirect", "safety_urgent",
        "medical_out_of_scope", "needs_clarification", "proceed".
    """
    text = message.strip()
    text_lower = text.lower()
    has_medical = _has_medical_signal(text_lower)

    # 1. Smalltalk — only when no medical signal words are present
    if not has_medical:
        for pattern in _SMALLTALK_PATTERNS:
            if pattern.search(text):
                return "smalltalk"

    # 2. Meta — questions about the assistant itself
    for pattern in _META_PATTERNS:
        if pattern.search(text):
            return "meta"

    # 3. Chitchat redirect — non-medical chatter (joke, weather, time …)
    if not has_medical:
        for pattern in _CHITCHAT_PATTERNS:
            if pattern.search(text):
                return "chitchat_redirect"

    # 4. Safety-urgent — emergency, self-diagnosis confirmation, self-treatment
    for pattern in _SAFETY_URGENT_PATTERNS:
        if pattern.search(text):
            return "safety_urgent"

    # 5. Medical out-of-scope (treatment, prognosis, self-diagnosis)
    has_treatment = any(kw in text_lower for kw in _TREATMENT_KEYWORDS)
    has_prognosis = any(kw in text_lower for kw in _PROGNOSIS_KEYWORDS)
    has_diagnosis = any(kw in text_lower for kw in _DIAGNOSIS_PHRASES)

    # Override: if the message also mentions referral-related context,
    # it is likely an NG12-relevant question and should proceed.
    has_referral_context = any(
        kw in text_lower for kw in _REFERRAL_CONTEXT_KEYWORDS
    )

    if (has_treatment or has_prognosis or has_diagnosis) and not has_referral_context:
        return "medical_out_of_scope"

    # 6. Needs clarification — vague symptoms without specific NG12 detail
    has_vague = any(p.search(text) for p in _VAGUE_SYMPTOM_PATTERNS)
    has_specific = any(kw in text_lower for kw in _SPECIFIC_NG12_SYMPTOMS)
    if has_vague and not has_specific:
        return "needs_clarification"

    return "proceed"


# ---------------------------------------------------------------------------
# Node 1b: input_guardrail (after load_history, before build_query)
# ---------------------------------------------------------------------------

async def input_guardrail_node(state: ChatState) -> dict:
    """Classify the message and short-circuit for smalltalk / meta queries."""
    classification = classify_input(state["message"])
    print(f"[InputGuardrail] Classification: {classification}")
    return {"guardrail_result": classification}


def route_input_guardrail(state: ChatState) -> str:
    """Return the input guardrail classification to select the next node."""
    return state["guardrail_result"]


# ---------------------------------------------------------------------------
# Node 1c: smalltalk_meta (returns canned response, skips RAG)
# ---------------------------------------------------------------------------

async def smalltalk_meta_node(state: ChatState) -> dict:
    """Return a canned response for smalltalk, meta, chitchat, safety, or clarification queries."""
    guardrail = state["guardrail_result"]
    if guardrail == "meta":
        answer = META_RESPONSE
    elif guardrail == "chitchat_redirect":
        answer = CHAT_NON_MEDICAL_REDIRECT_RESPONSE
    elif guardrail == "safety_urgent":
        answer = CHAT_SAFETY_RESPONSE
    elif guardrail == "needs_clarification":
        answer = CHAT_CLARIFY_RESPONSE
    else:
        answer = SMALLTALK_RESPONSE
    return {"answer": answer, "citations": [], "query_summary": None}


# ---------------------------------------------------------------------------
# Node 2: build_query
# ---------------------------------------------------------------------------

async def build_query_node(state: ChatState) -> dict:
    """Build a retrieval query using the tiered A+C+B strategy."""
    session_id = state["session_id"]
    topic = session_store.get_topic(session_id)
    print(f"[BuildQuery] Current topic for {session_id}: '{topic}'")

    search_query, query_strategy = await query_builder.build(
        session_id, state["message"],
    )
    logger.info(
        "Query strategy: %s, query: %s", query_strategy, search_query,
    )
    print(f"[Chat] Query strategy: {query_strategy}, query: {search_query}")
    return {"search_query": search_query, "query_strategy": query_strategy}


# ---------------------------------------------------------------------------
# Node 3: retrieve
# ---------------------------------------------------------------------------

async def retrieve_node(state: ChatState) -> dict:
    """Retrieve relevant NG12 guideline chunks for the query."""
    chunks = rag_pipeline.retrieve(state["search_query"], top_k=6)
    return {"chunks": chunks}


# ---------------------------------------------------------------------------
# Node 4: guardrail_check
# ---------------------------------------------------------------------------

def _assess_chunk_quality(chunks: list[dict]) -> str:
    """Rate retrieved chunks as 'sufficient', 'weak', or 'none'."""
    if not chunks:
        return "none"

    scores = [c.get("score", 0) for c in chunks]
    best = max(scores)

    # All scores below the floor -> nothing useful
    if all(s < 0.25 for s in scores):
        return "none"

    # Count how many chunks have a reasonably good score
    good_chunks = sum(1 for s in scores if s > 0.4)

    if good_chunks == 0:
        # No chunk above 0.4
        if best < 0.35:
            return "none"
        return "weak"

    if good_chunks <= 2 and best < 0.5:
        return "weak"

    return "sufficient"


def _has_lexical_overlap(message: str, chunks: list[dict]) -> bool:
    """Check whether any meaningful word in *message* appears in the chunks.

    Returns False when the query is completely unrelated to every chunk
    (e.g. "quantum physics" vs. clinical guideline text).
    """
    msg_words = set(message.lower().split()) - _STOP_WORDS
    if not msg_words:
        return True  # nothing meaningful to check -> assume OK

    for chunk in chunks:
        chunk_lower = chunk.get("text", "").lower()
        if any(w in chunk_lower for w in msg_words):
            return True
    return False


async def guardrail_check_node(state: ChatState) -> dict:
    """Check chunk quality and optionally retry with an LLM-rewritten query.

    Performs three checks in order:
    1. Out-of-scope detection (independent of chunk quality).
    2. Chunk quality assessment + lexical overlap guard.
    3. LLM rewrite retry when quality is 'none'.
    """
    chunks = state["chunks"] or []
    search_query = state["search_query"]
    query_strategy = state["query_strategy"]
    message = state["message"]
    msg_lower = message.lower()

    # === Step 1: out-of-scope detection (independent of chunks) ===
    has_oos = any(kw in msg_lower for kw in _OUT_OF_SCOPE_KEYWORDS)
    has_in_scope = any(kw in msg_lower for kw in _IN_SCOPE_KEYWORDS)

    # === Score metrics (computed once, reused below) ===
    def _build_score_breakdown(ch: list[dict]) -> dict:
        if ch:
            scores = [c.get("score", 0) for c in ch]
            return {
                "top1_score": round(scores[0], 3),
                "mean_score": round(sum(scores) / len(scores), 3),
                "above_035_count": len([s for s in scores if s > 0.35]),
                "total_chunks": len(ch),
            }
        return {
            "top1_score": 0,
            "mean_score": 0,
            "above_035_count": 0,
            "total_chunks": 0,
        }

    if has_oos and not has_in_scope:
        print(f"[Guardrail] Out of scope: '{message}'")
        return {
            "guardrail_result": "out_of_scope",
            "chunks": chunks,
            "search_query": search_query,
            "query_strategy": query_strategy,
            "score_breakdown": _build_score_breakdown(chunks),
        }

    # === Step 2: chunk quality assessment ===
    result = _assess_chunk_quality(chunks)

    print(
        f"[Guardrail] Result: {result}, "
        f"scores: {[round(c.get('score', 0), 3) for c in chunks]}"
    )

    # Lexical overlap guard: if no meaningful word appears in any chunk text,
    # downgrade to 'none' regardless of cosine score.  Use the search query
    # (which may include topic enrichment) rather than the short original
    # message, so that topic words like "lung" count as overlap.
    overlap_text = search_query if search_query else message
    if result in ("sufficient", "weak") and not _has_lexical_overlap(overlap_text, chunks):
        print(f"[Guardrail] No lexical overlap, downgrading to 'none'")
        result = "none"

    # === Step 3: retry with LLM rewrite if quality is 'none' ===
    if (
        result == "none"
        and query_strategy != "llm_rewrite"
        and gemini_client.is_available
    ):
        history = state.get("history") or []
        if history:
            history_text = format_history(history, max_turns=6)
            rewrite_prompt = REWRITE_PROMPT.format(
                history=history_text,
                message=message,
            )
            try:
                new_query = await gemini_client.generate(
                    system_prompt="",
                    user_prompt=rewrite_prompt,
                )
                if new_query and new_query.strip():
                    new_query = new_query.strip()
                    print(f"[Chat] Retry with LLM rewrite: {new_query}")
                    chunks = rag_pipeline.retrieve(new_query, top_k=6)
                    result = _assess_chunk_quality(chunks)
                    search_query = new_query
                    query_strategy = "llm_rewrite"
            except Exception as exc:
                logger.warning("LLM rewrite retry failed: %s", exc)

    return {
        "chunks": chunks,
        "search_query": search_query,
        "query_strategy": query_strategy,
        "guardrail_result": result,
        "score_breakdown": _build_score_breakdown(chunks),
    }


# ---------------------------------------------------------------------------
# Node 4b: summarize_query (display-only, does NOT affect retrieval)
# ---------------------------------------------------------------------------

async def summarize_query_node(state: ChatState) -> dict:
    """Extract key clinical information from user message for display.

    This node runs AFTER guardrail_check and BEFORE generate/qualify.
    The summary is prepended to the final answer for transparency but
    has no effect on the retrieval query or chunk scoring.

    Includes conversation history so that clinical details mentioned in
    earlier turns (e.g. age, gender, prior symptoms) are carried forward.
    """
    message = state["message"]
    history = state.get("history") or []
    guardrail_result = state.get("guardrail_result")

    # Skip summarization for non-clinical query paths
    if guardrail_result in ("smalltalk", "meta", "chitchat_redirect", "out_of_scope", "none"):
        return {"query_summary": None}

    # Use Gemini to extract structured info
    if gemini_client and gemini_client.is_available:
        # Format recent USER messages only — assistant responses contain
        # guideline thresholds (e.g. "45 or over") that the LLM can
        # incorrectly extract as patient details.
        history_context = ""
        if history:
            user_msgs = [m for m in history[-8:] if m["role"] == "user"]
            if user_msgs:
                history_context = "Previous user messages:\n"
                for msg in user_msgs:
                    content = msg["content"]
                    if len(content) > 200:
                        content = content[:200] + "..."
                    history_context += f"- {content}\n"
                history_context += "\n"

        extract_prompt = (
            "Extract key clinical information from the conversation "
            "below.\n"
            "Include details from BOTH the previous conversation AND "
            "the current question.\n"
            "If the user mentioned age, gender, or symptoms in an "
            "earlier message, carry those forward.\n\n"
            "STRICT RULES:\n"
            "- ONLY include information explicitly stated by the user "
            "somewhere in this conversation\n"
            "- Do NOT infer, guess, or hallucinate details never "
            "mentioned\n"
            "- Do NOT use general medical knowledge to fill gaps\n"
            "- If a field was NOT mentioned anywhere in the "
            "conversation, write [None]\n"
            "- Include symptoms or conditions the user is ASKING "
            "ABOUT, not only symptoms they claim to have personally\n"
            "- Include hypothetical ages or scenarios the user "
            "mentions (e.g. 'under 40', 'if I'm a smoker')\n\n"
            f"{history_context}"
            f"Current question: {message}\n\n"
            "Return a brief structured summary in this exact format:\n\n"
            "Patient details: [any age, gender, or risk factors "
            "mentioned or asked about, otherwise None]\n"
            "Symptoms: [any symptoms or conditions mentioned or "
            "asked about, otherwise None]\n"
            "Duration/timing: [if mentioned anywhere in "
            "conversation, otherwise None]\n"
            "Question: [what they're asking now]\n\n"
            "Keep each field to one line, under 20 words.\n\n"
            "Summary:"
        )

        try:
            summary = await gemini_client.generate(
                system_prompt=(
                    "You extract clinical information from conversations. "
                    "Carry forward details from earlier messages. "
                    "ONLY report what was explicitly stated by the user. "
                    "Never infer or hallucinate details not mentioned. "
                    "Use [None] for fields not mentioned anywhere."
                ),
                user_prompt=extract_prompt,
            )
            result = summary.strip() if summary else None
            print(f"[Summarize] Extracted summary: {result}")
            return {"query_summary": result}
        except Exception as e:
            logger.warning("Query summarization failed: %s", e)
            print(f"[Summarize] Failed: {e}")
            return {"query_summary": None}

    return {"query_summary": None}


def route_after_summarize(state: ChatState) -> str:
    """Route from summarize_query to generate or qualify based on guardrail_result."""
    return state["guardrail_result"]


# ---------------------------------------------------------------------------
# Routing function for conditional edges
# ---------------------------------------------------------------------------

def route_guardrail(state: ChatState) -> str:
    """Return the guardrail result to select the next node."""
    return state["guardrail_result"]


# ---------------------------------------------------------------------------
# Shared helper: generate an answer from Gemini or produce a demo fallback
# ---------------------------------------------------------------------------

async def _generate_answer(
    message: str,
    chunks: list[dict],
    history: list[dict],
) -> str:
    """Call Gemini to generate a grounded answer, or return a demo answer."""
    if gemini_client.is_available:
        prompt = format_chat_prompt(message, chunks, history)
        answer = await gemini_client.generate(
            system_prompt=CHAT_SYSTEM_PROMPT,
            user_prompt=prompt,
        )
        if answer:
            return answer

    # Demo fallback when Gemini is unavailable
    demo_answer = "Demo mode - Gemini not configured.\n\nRelevant guidelines found:\n"
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        demo_answer += (
            f"\n[Source {i}] Section {meta.get('section', 'N/A')}"
            f" ({meta.get('action_type', 'N/A')}): "
            f"{chunk['text'][:150]}...\n"
        )
    return demo_answer


# ---------------------------------------------------------------------------
# Node 5: generate (guardrail_result == "sufficient")
# ---------------------------------------------------------------------------

async def generate_node(state: ChatState) -> dict:
    """Generate a full grounded answer from retrieved chunks."""
    chunks = state["chunks"] or []
    history = state.get("history") or []
    message = state["message"]

    answer = await _generate_answer(message, chunks, history)
    citations = build_citations_from_chunks(chunks, answer)
    answer = clean_answer_sources(answer, chunks)

    # Prepend query summary for display — only for clinical paths
    guardrail = state.get("guardrail_result")
    if (
        guardrail in ("sufficient", "weak")
        and state.get("query_summary")
    ):
        summary_display = (
            f"\U0001f4cb **Understanding your question:**\n"
            f"{state['query_summary']}\n\n---\n\n"
        )
        answer = summary_display + answer

    # Citation validation: append a transparency note when an LLM-generated
    # answer has no citations to back it up.
    if gemini_client.is_available and not citations and len(answer) > 50:
        answer = answer + (
            "\n\n_Note: I was unable to provide specific guideline "
            "citations for this response. Please verify this information "
            "against the NG12 guideline directly._"
        )

    return {"answer": answer, "citations": citations, "citation_count": len(citations)}


# ---------------------------------------------------------------------------
# Node 6: qualify (guardrail_result == "weak")
# ---------------------------------------------------------------------------

async def qualify_node(state: ChatState) -> dict:
    """Generate a qualified / hedged answer from weak evidence."""
    chunks = state["chunks"] or []
    history = state.get("history") or []
    message = state["message"]

    partial_answer = await _generate_answer(message, chunks, history)
    citations = build_citations_from_chunks(chunks, partial_answer)
    partial_answer = clean_answer_sources(partial_answer, chunks)
    answer = CHAT_QUALIFY_TEMPLATE.format(partial_answer=partial_answer)

    # Prepend query summary for display — only for clinical paths
    guardrail = state.get("guardrail_result")
    if (
        guardrail in ("sufficient", "weak")
        and state.get("query_summary")
    ):
        summary_display = (
            f"\U0001f4cb **Understanding your question:**\n"
            f"{state['query_summary']}\n\n---\n\n"
        )
        answer = summary_display + answer

    # Citation validation: append a transparency note when an LLM-generated
    # answer has no citations to back it up.
    if gemini_client.is_available and not citations and len(answer) > 50:
        answer = answer + (
            "\n\n_Note: I was unable to provide specific guideline "
            "citations for this response. Please verify this information "
            "against the NG12 guideline directly._"
        )

    return {"answer": answer, "citations": citations, "citation_count": len(citations)}


# ---------------------------------------------------------------------------
# Node 7: refuse (guardrail_result == "none")
# ---------------------------------------------------------------------------

async def refuse_node(state: ChatState) -> dict:
    """Return a refusal when no relevant evidence was found."""
    return {"answer": CHAT_REFUSE_RESPONSE, "citations": []}


# ---------------------------------------------------------------------------
# Node 8: out_of_scope (guardrail_result == "out_of_scope")
# ---------------------------------------------------------------------------

async def out_of_scope_node(state: ChatState) -> dict:
    """Return an out-of-scope message for non-NG12 topics."""
    return {"answer": CHAT_OUT_OF_SCOPE_RESPONSE, "citations": []}


# ---------------------------------------------------------------------------
# Node 9: save_history
# ---------------------------------------------------------------------------

async def save_history_node(state: ChatState) -> dict:
    """Persist the current turn in the session store and update the topic.

    Topic is only updated when guardrail_result is 'sufficient' or 'weak'
    AND there are actual citations.  Only cited chunks are used for the
    topic update to prevent drift from irrelevant retrieved chunks.
    """
    session_id = state["session_id"]
    message = state["message"]
    answer = state.get("answer", "")
    chunks = state.get("chunks") or []
    citations = state.get("citations") or []
    guardrail_result = state.get("guardrail_result", "")

    # --- DEBUG: trace session store instance and state ---
    print(f"[SaveHistory][DEBUG] session_id={session_id}")
    print(f"[SaveHistory][DEBUG] message={message!r}")
    print(f"[SaveHistory][DEBUG] answer length={len(answer)}")
    print(f"[SaveHistory][DEBUG] session_store id={id(session_store)}")
    print(f"[SaveHistory][DEBUG] sessions keys BEFORE append: {list(session_store._sessions.keys())}")
    history_before = session_store.get_history(session_id)
    print(f"[SaveHistory][DEBUG] history count BEFORE append: {len(history_before)}")

    session_store.append(session_id, "user", message)
    session_store.append(session_id, "assistant", answer)

    history_after = session_store.get_history(session_id)
    print(f"[SaveHistory][DEBUG] history count AFTER append: {len(history_after)}")
    print(f"[SaveHistory][DEBUG] sessions keys AFTER append: {list(session_store._sessions.keys())}")
    # --- END DEBUG ---

    if chunks and citations and guardrail_result in ("sufficient", "weak"):
        # Filter chunks to only those actually cited in the answer.
        # Match by chunk_id since [Source N] markers have already been
        # replaced by clean_answer_sources() before this node runs.
        cited_ids = {c.get("chunk_id") for c in citations}
        cited_chunks = [
            c for c in chunks
            if c.get("metadata", {}).get("chunk_id") in cited_ids
        ]
        if cited_chunks:
            session_store.update_topic(session_id, cited_chunks)

    print(
        f"[SaveHistory] Topic after update: "
        f"'{session_store.get_topic(session_id)}'"
    )
    # LangGraph 0.2.x requires nodes to return at least one state key
    return {"history": session_store.get_history(session_id)}


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------

def build_chat_graph():
    """Construct the LangGraph StateGraph for conversational chat.

    Returns:
        A compiled LangGraph graph.
    """
    graph = StateGraph(ChatState)

    # Add nodes
    graph.add_node("load_history", load_history_node)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("smalltalk_meta", smalltalk_meta_node)
    graph.add_node("build_query", build_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("guardrail_check", guardrail_check_node)
    graph.add_node("summarize_query", summarize_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("qualify", qualify_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("save_history", save_history_node)

    # Entry point
    graph.set_entry_point("load_history")

    # load_history -> input_guardrail (classify first)
    graph.add_edge("load_history", "input_guardrail")

    # Conditional branch after input_guardrail
    graph.add_conditional_edges(
        "input_guardrail",
        route_input_guardrail,
        {
            "smalltalk": "smalltalk_meta",
            "meta": "smalltalk_meta",
            "chitchat_redirect": "smalltalk_meta",
            "safety_urgent": "smalltalk_meta",
            "needs_clarification": "smalltalk_meta",
            "medical_out_of_scope": "out_of_scope",
            "proceed": "build_query",
        },
    )

    # Linear edges for the RAG path
    graph.add_edge("build_query", "retrieve")
    graph.add_edge("retrieve", "guardrail_check")

    # Conditional branch after guardrail: sufficient/weak go through
    # summarize_query first; none/out_of_scope skip directly.
    graph.add_conditional_edges(
        "guardrail_check",
        route_guardrail,
        {
            "sufficient": "summarize_query",
            "weak": "summarize_query",
            "none": "refuse",
            "out_of_scope": "out_of_scope",
        },
    )

    # After summarize_query, route to generate or qualify based on
    # the original guardrail_result (summarize_query does not change it).
    graph.add_conditional_edges(
        "summarize_query",
        route_after_summarize,
        {
            "sufficient": "generate",
            "weak": "qualify",
        },
    )

    # All response nodes converge to save_history
    graph.add_edge("smalltalk_meta", "save_history")
    graph.add_edge("generate", "save_history")
    graph.add_edge("qualify", "save_history")
    graph.add_edge("refuse", "save_history")
    graph.add_edge("out_of_scope", "save_history")
    graph.add_edge("save_history", END)

    return graph.compile()


# Compiled workflow singleton
chat_workflow = build_chat_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_chat(session_id: str, message: str) -> dict:
    """Execute the chat workflow for a user message.

    Args:
        session_id: The session identifier for conversation continuity.
        message: The user's message text.

    Returns:
        Dict with keys: session_id, answer, citations,
        query_strategy, search_query, guardrail_result.
    """
    initial_state: ChatState = {
        "session_id": session_id,
        "message": message,
        "history": None,
        "search_query": None,
        "query_strategy": None,
        "chunks": None,
        "answer": None,
        "citations": None,
        "citation_count": None,
        "score_breakdown": None,
        "guardrail_result": None,
        "query_summary": None,
    }

    result = await chat_workflow.ainvoke(initial_state)

    return {
        "session_id": session_id,
        "answer": result.get("answer", "An error occurred."),
        "citations": result.get("citations", []),
        "debug": {
            "query_strategy": result.get("query_strategy"),
            "search_query": result.get("search_query"),
            "guardrail_result": result.get("guardrail_result"),
            "citation_count": result.get("citation_count", 0),
            "score_breakdown": result.get("score_breakdown"),
            "query_summary": result.get("query_summary"),
        },
    }
