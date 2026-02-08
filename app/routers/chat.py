"""
Chat Router

POST   /chat                       - Conversational Q&A with NG12 guidelines
GET    /chat/history/{session_id}   - Retrieve conversation history for a session
DELETE /chat/history/{session_id}   - Clear a session's history and topic
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.agents.chat_workflow import run_chat
from app.agents import chat_workflow as _chat_wf_module
from app.memory.session_store import session_store
from app.models.schemas import ChatRequest, ChatResponse, Citation

router = APIRouter()


# ── POST /chat ────────────────────────────────────────────────────────────────

@router.post("")
async def chat(request: ChatRequest) -> dict[str, Any]:
    """Handle a conversational chat message.

    Returns a ChatResponse-compatible dict augmented with a ``debug`` field
    containing query strategy, search query, and guardrail result.
    """
    result = await run_chat(request.session_id, request.message)
    debug = result.get("debug", {})

    return {
        "session_id": result["session_id"],
        "answer": result["answer"],
        "citations": [
            Citation(**c).model_dump() for c in result["citations"]
        ],
        "debug": {
            "query_strategy": debug.get("query_strategy"),
            "search_query": debug.get("search_query"),
            "guardrail_result": debug.get("guardrail_result"),
            "citation_count": debug.get("citation_count", 0),
            "score_breakdown": debug.get("score_breakdown"),
        },
    }


# ── GET /chat/history/{session_id} ────────────────────────────────────────────

@router.get("/history/{session_id}")
async def get_history(session_id: str) -> dict[str, Any]:
    """Retrieve conversation history and topic for a session."""
    # --- DEBUG: compare session_store instances ---
    workflow_store = _chat_wf_module.session_store
    print(f"[GetHistory][DEBUG] router   session_store id={id(session_store)}")
    print(f"[GetHistory][DEBUG] workflow session_store id={id(workflow_store)}")
    print(f"[GetHistory][DEBUG] same instance? {session_store is workflow_store}")
    print(f"[GetHistory][DEBUG] router   _sessions keys: {list(session_store._sessions.keys())}")
    print(f"[GetHistory][DEBUG] workflow _sessions keys: {list(workflow_store._sessions.keys())}")
    # --- END DEBUG ---
    history = session_store.get_history(session_id)
    topic = session_store.get_topic(session_id)
    return {
        "session_id": session_id,
        "history": history,
        "topic": topic,
    }


# ── DELETE /chat/history/{session_id} ─────────────────────────────────────────

@router.delete("/history/{session_id}")
async def clear_history(session_id: str) -> dict[str, str]:
    """Clear a single session's history and topic."""
    session_store.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}
