"""
Admin Router

Endpoints for managing and inspecting the ChromaDB vector store:
  POST /admin/refresh           - Re-ingest the NG12 PDF
  GET  /admin/stats             - Collection statistics
  GET  /admin/chunks            - Paginated chunk listing with filters
  GET  /admin/chunks/{chunk_id} - Single chunk detail with embedding preview
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.core import vector_store
from app.ingestion.ingest import ingest_ng12
from app.memory.session_store import session_store
from app.models.schemas import RefreshResponse

router = APIRouter()


# ── POST /admin/refresh ─────────────────────────────────────────────────────

@router.post("/refresh", response_model=RefreshResponse)
async def refresh() -> RefreshResponse:
    """Re-index the NG12 PDF and clear chat sessions."""
    count = ingest_ng12(settings.PDF_PATH)
    session_store.clear_all()
    return RefreshResponse(
        status="success",
        chunks_indexed=count,
        canonical_stored=vector_store.count_canonical(),
        sessions_cleared=True,
    )


# ── GET /admin/stats ─────────────────────────────────────────────────────────

@router.get("/stats")
async def stats() -> dict[str, Any]:
    """Return aggregate statistics across both collections."""
    # -- Search collection stats --
    search_data = vector_store.get_all()
    search_total = len(search_data["ids"])

    doc_type_counter: Counter[str] = Counter()
    system_title_counter: Counter[str] = Counter()

    for meta in search_data["metadatas"]:
        dt = meta.get("doc_type", "unknown")
        doc_type_counter[dt] += 1
        if dt == "symptom_index":
            st = meta.get("system_title", "Unknown")
            system_title_counter[st] += 1

    # -- Canonical collection stats --
    canonical_chunks = vector_store.list_canonical()
    canonical_total = len(canonical_chunks)

    cancer_type_counter: Counter[str] = Counter()
    action_type_counter: Counter[str] = Counter()
    urgency_counter: Counter[str] = Counter()
    chunks_with_age = 0
    chunks_with_symptoms = 0

    for c in canonical_chunks:
        meta = c["metadata"]
        cancer_type_counter[meta.get("cancer_type", "Unknown")] += 1
        action_type_counter[meta.get("action_type", "Other")] += 1
        urgency_counter[meta.get("urgency", "none")] += 1

        if meta.get("age_min") is not None or meta.get("age_max") is not None:
            chunks_with_age += 1
        if meta.get("symptom_keywords_json"):
            chunks_with_symptoms += 1

    return {
        "search_collection_total": search_total,
        "canonical_collection_total": canonical_total,
        "total_chunks": search_total + canonical_total,
        "doc_type_distribution": dict(doc_type_counter),
        "cancer_type_distribution": dict(cancer_type_counter),
        "action_type_distribution": dict(action_type_counter),
        "urgency_distribution": dict(urgency_counter),
        "system_title_distribution": dict(system_title_counter),
        "chunks_with_age_threshold": chunks_with_age,
        "chunks_with_symptoms": chunks_with_symptoms,
    }


# ── GET /admin/chunks ────────────────────────────────────────────────────────

@router.get("/chunks")
async def list_chunks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    collection: str = Query("search"),
    doc_type: Optional[str] = Query(None),
    cancer_type: Optional[str] = Query(None),
    action_type: Optional[str] = Query(None),
    system_title: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Return a paginated, filterable list of chunks.

    Use ``collection=search`` (default) for the main search collection
    or ``collection=canonical`` for the canonical rules collection.
    """
    items: list[dict[str, Any]] = []

    if collection == "canonical":
        all_chunks = vector_store.list_canonical()
        for c in all_chunks:
            meta = c["metadata"]
            text = c["text"]
            symptoms: list[str] = []
            skj = meta.get("symptom_keywords_json")
            if skj:
                try:
                    symptoms = json.loads(skj)
                except (json.JSONDecodeError, TypeError):
                    pass
            items.append({
                "chunk_id": c["chunk_id"],
                "doc_type": "rule_canonical",
                "section": meta.get("section", ""),
                "cancer_type": meta.get("cancer_type", ""),
                "action_type": meta.get("action_type", ""),
                "urgency": meta.get("urgency", ""),
                "page": meta.get("page"),
                "page_end": meta.get("page_end"),
                "age_min": meta.get("age_min"),
                "age_max": meta.get("age_max"),
                "symptom_keywords": symptoms,
                "text": text,
                "text_preview": text[:150] if text else "",
            })
    else:
        # Default: search collection
        data = vector_store.get_all()
        for i, chunk_id in enumerate(data["ids"]):
            meta = data["metadatas"][i]
            text = data["documents"][i]
            items.append({
                "chunk_id": chunk_id,
                "doc_type": meta.get("doc_type", ""),
                "section": meta.get("section", ""),
                "rule_id": meta.get("rule_id", ""),
                "cancer_type": meta.get("cancer_type", ""),
                "system_title": meta.get("system_title", ""),
                "sub_title": meta.get("sub_title", ""),
                "symptom": meta.get("symptom", ""),
                "possible_cancer": meta.get("possible_cancer", ""),
                "references_json": meta.get("references_json", ""),
                "action_type": meta.get("action_type", ""),
                "page": meta.get("page"),
                "page_end": meta.get("page_end"),
                "text": text,
                "text_preview": text[:150] if text else "",
            })

    # ── Filtering ────────────────────────────────────────────────────────
    if doc_type:
        items = [c for c in items if c.get("doc_type") == doc_type]
    if cancer_type:
        items = [c for c in items if c.get("cancer_type") == cancer_type]
    if action_type:
        items = [c for c in items if c.get("action_type") == action_type]
    if system_title:
        items = [c for c in items if c.get("system_title") == system_title]
    if search:
        needle = search.lower()
        items = [c for c in items if needle in c.get("text", "").lower()]

    # ── Pagination ───────────────────────────────────────────────────────
    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "chunks": page_items,
    }


# ── GET /admin/chunks/{chunk_id} ─────────────────────────────────────────────

@router.get("/chunks/{chunk_id}")
async def get_chunk(chunk_id: str) -> dict[str, Any]:
    """Return full detail for a single chunk including embedding preview."""
    result = vector_store.get_by_id(chunk_id)
    if result is None:
        return {"error": "Chunk not found", "chunk_id": chunk_id}
    return result


# ── GET /admin/canonical ────────────────────────────────────────────────────

@router.get("/canonical")
async def list_canonical_rules() -> dict[str, Any]:
    """Return all canonical chunks for admin inspection."""
    chunks = vector_store.list_canonical()
    return {
        "count": len(chunks),
        "chunks": [
            {
                "chunk_id": c["chunk_id"],
                "section": c["metadata"].get("section"),
                "cancer_type": c["metadata"].get("cancer_type"),
                "action_type": c["metadata"].get("action_type"),
                "urgency": c["metadata"].get("urgency"),
                "text": c["text"],
            }
            for c in chunks
        ],
    }


# ── GET /admin/canonical/{rule_id} ─────────────────────────────────────────

@router.get("/canonical/{rule_id}")
async def get_canonical_rule(rule_id: str) -> dict[str, Any]:
    """Return a single canonical chunk by rule_id (e.g. '1.1.1')."""
    result = vector_store.get_canonical(rule_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Rule {rule_id} not found")
    return result
