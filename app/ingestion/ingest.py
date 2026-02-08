"""
PDF Ingestion Script

Parses the NG12 clinical guideline PDF and builds the ChromaDB vector index.
Can be run standalone: python -m app.ingestion.ingest
"""

from app.config import settings
from app.core import vector_store
from app.ingestion.chunker import chunk_ng12, parse_pdf_to_lines

INDEXABLE_TYPES = {"rule_search", "symptom_index"}


def ingest_ng12(pdf_path: str) -> int:
    """Parse the NG12 PDF and index its chunks into ChromaDB.

    Pipeline:
      1. parse_pdf_to_lines - extract and clean text lines from PDF
      2. chunk_ng12 - split into structured recommendation chunks
      3. Separate canonical chunks from indexable chunks
      4. vector_store.reset - clear both collections
      5. vector_store.add_chunks - embed and store search chunks
      6. vector_store.add_canonical_chunks - store canonical chunks

    Args:
        pdf_path: Path to the NG12 guideline PDF file.

    Returns:
        Total number of chunks processed.
    """
    print(f"Parsing PDF: {pdf_path}")
    lines = parse_pdf_to_lines(pdf_path)
    print(f"Extracted {len(lines)} cleaned lines")

    chunks = chunk_ng12(lines)

    # Separate canonical vs indexable chunks
    canonical_chunks = [
        c for c in chunks
        if c["metadata"].get("doc_type") == "rule_canonical"
    ]
    index_chunks = [
        c for c in chunks
        if c["metadata"].get("doc_type") in INDEXABLE_TYPES
    ]

    print("\nResetting vector store...")
    vector_store.reset()

    print(f"Indexing {len(index_chunks)} search chunks into ChromaDB...")
    vector_store.add_chunks(index_chunks)

    print(f"Indexing {len(canonical_chunks)} canonical chunks into ChromaDB...")
    vector_store.add_canonical_chunks(canonical_chunks)

    # Print write summary
    search_count = len([
        c for c in index_chunks
        if c["metadata"]["doc_type"] == "rule_search"
    ])
    symptom_count = len([
        c for c in index_chunks
        if c["metadata"]["doc_type"] == "symptom_index"
    ])
    print(f"\nWrite summary:")
    print(f"  Search collection (ng12_guidelines): {vector_store.count()} docs")
    print(f"    - rule_search: {search_count}")
    print(f"    - symptom_index: {symptom_count}")
    print(f"  Canonical collection (ng12_canonical): {vector_store.count_canonical()} docs")

    return len(chunks)


if __name__ == "__main__":
    count = ingest_ng12(settings.PDF_PATH)
    print(f"\nIngestion complete. Processed {count} total chunks.")
