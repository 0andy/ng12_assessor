"""
ChromaDB Vector Store

Manages the NG12 guideline vector index.
Uses ChromaDB PersistentClient with default embeddings.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import chromadb

from app.config import settings

COLLECTION_NAME = "ng12_guidelines"
CANONICAL_COLLECTION_NAME = "ng12_canonical"

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None
_canonical_collection: Optional[chromadb.Collection] = None


def _get_client() -> chromadb.PersistentClient:
    """Lazy-initialize the ChromaDB persistent client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    return _client


def get_or_create_collection() -> chromadb.Collection:
    """Return the ChromaDB collection, creating it if necessary."""
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_or_create_canonical_collection() -> chromadb.Collection:
    """Return the canonical ChromaDB collection, creating it if necessary."""
    global _canonical_collection
    if _canonical_collection is None:
        client = _get_client()
        _canonical_collection = client.get_or_create_collection(
            name=CANONICAL_COLLECTION_NAME,
        )
    return _canonical_collection


def add_chunks(chunks: list[dict[str, Any]]) -> int:
    """Add document chunks to the vector store.

    Handles ChromaDB metadata constraints:
      - Removes None values from metadata (ChromaDB rejects them)
      - list-type values should already be JSON-serialized by the chunker

    Args:
        chunks: List of dicts with keys: chunk_id, text, metadata.

    Returns:
        Number of chunks indexed.
    """
    collection = get_or_create_collection()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        # Filter out None values from metadata
        clean_meta = {
            k: v for k, v in chunk["metadata"].items() if v is not None
        }
        metadatas.append(clean_meta)

    # ChromaDB supports batched upsert; use upsert to be idempotent
    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return len(ids)


def query(query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Query the vector store for relevant chunks.

    ChromaDB uses cosine distance by default:
      distance = 1 - cosine_similarity
      score = 1 - distance  (range 0..1, 1 = most similar)

    Converts symptom_keywords_json back to a list in the returned metadata.

    Args:
        query_text: The search query.
        top_k: Number of results to return.

    Returns:
        List of result dicts with keys: chunk_id, text, metadata, score.
    """
    collection = get_or_create_collection()

    results = collection.query(
        query_texts=[query_text],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if not results["ids"] or not results["ids"][0]:
        return output

    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]
        score = 1.0 - distance  # cosine similarity

        meta = dict(results["metadatas"][0][i])
        # Convert symptom_keywords_json back to list
        if "symptom_keywords_json" in meta:
            try:
                meta["symptom_keywords"] = json.loads(
                    meta["symptom_keywords_json"]
                )
            except (json.JSONDecodeError, TypeError):
                meta["symptom_keywords"] = []

        output.append({
            "chunk_id": doc_id,
            "text": results["documents"][0][i],
            "metadata": meta,
            "score": score,
        })

    return output


def add_canonical_chunks(chunks: list[dict[str, Any]]) -> int:
    """Write canonical chunks to the ng12_canonical collection.

    Args:
        chunks: List of dicts with keys: chunk_id, text, metadata.

    Returns:
        Number of chunks indexed.
    """
    collection = get_or_create_canonical_collection()

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        ids.append(chunk["chunk_id"])
        documents.append(chunk["text"])
        # Filter out None values from metadata
        clean_meta = {
            k: v for k, v in chunk["metadata"].items() if v is not None
        }
        metadatas.append(clean_meta)

    batch_size = 100
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return len(ids)


def get_canonical(rule_id: str) -> dict[str, Any] | None:
    """Look up a canonical chunk by rule_id (e.g. "1.1.1").

    Args:
        rule_id: The rule identifier such as "1.1.1".

    Returns:
        Dict with keys: chunk_id, text, metadata, or None if not found.
    """
    collection = get_or_create_canonical_collection()
    chunk_id = "ng12_" + rule_id.replace(".", "_")
    results = collection.get(ids=[chunk_id], include=["documents", "metadatas"])
    if results and results["ids"]:
        return {
            "chunk_id": results["ids"][0],
            "text": results["documents"][0],
            "metadata": results["metadatas"][0],
        }
    return None


def list_canonical() -> list[dict[str, Any]]:
    """Return all canonical chunks (for admin page).

    Returns:
        List of dicts with keys: chunk_id, text, metadata.
    """
    collection = get_or_create_canonical_collection()
    results = collection.get(include=["documents", "metadatas"])
    chunks = []
    for i, cid in enumerate(results["ids"]):
        chunks.append({
            "chunk_id": cid,
            "text": results["documents"][i],
            "metadata": results["metadatas"][i],
        })
    return chunks


def count_canonical() -> int:
    """Return the number of documents in the canonical collection."""
    collection = get_or_create_canonical_collection()
    return collection.count()


def reset_canonical() -> None:
    """Delete and recreate the canonical collection."""
    global _canonical_collection
    client = _get_client()
    try:
        client.delete_collection(name=CANONICAL_COLLECTION_NAME)
    except (ValueError, KeyError):
        pass
    _canonical_collection = None
    get_or_create_canonical_collection()


def reset() -> None:
    """Delete and recreate both collections (search + canonical)."""
    global _collection
    client = _get_client()
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except (ValueError, KeyError):
        pass  # Collection does not exist or DB schema is stale
    _collection = None
    get_or_create_collection()
    reset_canonical()


def count() -> int:
    """Return the number of documents in the collection."""
    collection = get_or_create_collection()
    return collection.count()


def get_all() -> dict[str, Any]:
    """Retrieve all chunks with documents and metadata.

    Returns:
        Dict with keys: ids, documents, metadatas.
        symptom_keywords_json is decoded back to a list in each metadata entry.
    """
    collection = get_or_create_collection()
    results = collection.get(include=["documents", "metadatas"])

    # Decode symptom_keywords_json back to list
    for meta in results["metadatas"]:
        if "symptom_keywords_json" in meta:
            try:
                meta["symptom_keywords"] = json.loads(
                    meta["symptom_keywords_json"]
                )
            except (json.JSONDecodeError, TypeError):
                meta["symptom_keywords"] = []

    return {
        "ids": results["ids"],
        "documents": results["documents"],
        "metadatas": results["metadatas"],
    }


def get_by_id(chunk_id: str) -> dict[str, Any] | None:
    """Retrieve a single chunk by ID with document, metadata, and embedding preview.

    Args:
        chunk_id: The unique identifier of the chunk.

    Returns:
        Dict with keys: chunk_id, text, metadata, embedding_preview (first 10 dims),
        or None if not found.
    """
    collection = get_or_create_collection()
    results = collection.get(
        ids=[chunk_id],
        include=["documents", "metadatas", "embeddings"],
    )

    if not results["ids"]:
        return None

    meta = dict(results["metadatas"][0])
    if "symptom_keywords_json" in meta:
        try:
            meta["symptom_keywords"] = json.loads(
                meta["symptom_keywords_json"]
            )
        except (json.JSONDecodeError, TypeError):
            meta["symptom_keywords"] = []

    embedding = results["embeddings"][0] if results["embeddings"] else []
    embedding_preview = embedding[:10] if embedding else []

    return {
        "chunk_id": results["ids"][0],
        "text": results["documents"][0],
        "metadata": meta,
        "embedding_preview": embedding_preview,
    }
