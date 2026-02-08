"""
NG12 Cancer Risk Assessor - FastAPI Application Entry Point

Registers routers for assessment, chat, and admin endpoints.
Serves the static frontend from the /static directory.
Auto-ingests the NG12 PDF on startup if the vector store is empty.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve .env relative to project root (parent of app/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers import assess, chat, admin

app = FastAPI(title="NG12 Cancer Risk Assessor")

# CORS - allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(assess.router, prefix="/assess", tags=["assess"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])

# Serve static files (index.html) at root - must be last so it doesn't
# shadow API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.on_event("startup")
async def startup_event():
    """Auto-ingest the NG12 PDF if the vector store is empty."""
    try:
        from app.core import vector_store
        from app.config import settings
        from app.ingestion.ingest import ingest_ng12

        search_count = vector_store.count()
        canonical_count = vector_store.count_canonical()
        if search_count == 0 or canonical_count == 0:
            print("One or both collections are empty. Running initial ingestion...")
            ingest_ng12(settings.PDF_PATH)
        else:
            print(f"Search collection: {search_count} documents")
            print(f"Canonical collection: {canonical_count} documents")
    except Exception as e:
        print(f"[Startup] Vector store initialization failed: {e}")
        print("[Startup] Continuing without vector store (chat history debug still works)")

    print("NG12 Assessor ready")
