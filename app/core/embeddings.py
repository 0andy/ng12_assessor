"""
Vertex AI Embeddings

Provides an embedding function for ChromaDB backed by Vertex AI.
Falls back to ChromaDB's default embedding when Google Cloud credentials
are not configured.
"""

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def get_embedding_function() -> Any:
    """Return a Vertex AI embedding function compatible with ChromaDB.

    If Google Cloud credentials are not configured, returns None so that
    ChromaDB can use its built-in default embedding model instead.

    Returns:
        A ChromaDB-compatible embedding function, or None.
    """
    if not settings.GOOGLE_CLOUD_PROJECT:
        logger.info(
            "GOOGLE_CLOUD_PROJECT not set - using ChromaDB default embeddings"
        )
        return None

    try:
        from chromadb.utils.embedding_functions import (
            GoogleVertexEmbeddingFunction,
        )

        ef = GoogleVertexEmbeddingFunction(
            project_id=settings.GOOGLE_CLOUD_PROJECT,
            region=settings.GOOGLE_CLOUD_LOCATION,
            model_name="text-embedding-004",
        )
        logger.info("Using Vertex AI embeddings (text-embedding-004)")
        return ef
    except Exception as exc:
        logger.warning("Failed to initialize Vertex AI embeddings: %s", exc)
        logger.info("Falling back to ChromaDB default embeddings")
        return None


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text string.

    Uses the Vertex AI embedding model when available, otherwise returns
    an empty list (callers should use ChromaDB's built-in embedding).

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the embedding vector, or empty list.
    """
    if not settings.GOOGLE_CLOUD_PROJECT:
        return []

    try:
        from vertexai.language_models import TextEmbeddingModel

        model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        embeddings = model.get_embeddings([text])
        return embeddings[0].values
    except Exception as exc:
        logger.warning("embed_text failed: %s", exc)
        return []
