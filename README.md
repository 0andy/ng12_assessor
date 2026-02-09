# NG12 Cancer Risk Assessor

A clinical decision support system that uses RAG to assess cancer risk based on NICE NG12 guidelines.

Live demo: <https://ng12assessor.fanai.dev/>

## Quick Start

### 1. Add your Google Cloud credentials

Place your GCP service account JSON key file in the project root, then:

```bash
cp .env.example .env
```

Edit `.env` and set the filename to match your key file:

```env
GOOGLE_APPLICATION_CREDENTIALS=your-credentials.json
```

The project ID is read automatically from the JSON key file.

### 2a. Run with Docker (recommended)

```bash
docker compose up --build
```

### 2b. Run locally

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

Then open <http://localhost:8000>

### 3. Vector Database

On first startup, the system automatically detects whether the ChromaDB vector store exists and builds it from the NG12 PDF if needed. No manual action is required.

To re-index manually, go to the **Vector DB Admin** tab and click **"Re-index PDF"**.

## Tech Stack

| Technology | Purpose |
|------------|---------|
| FastAPI | Backend framework with async support |
| LangGraph | Workflow orchestration (state machines) |
| ChromaDB | Vector database (2 collections) |
| Vertex AI | text-embedding-004 for embeddings |
| Gemini 2.0 Flash | LLM for reasoning & assessment |
| PyMuPDF | PDF parsing & text extraction |

## Design Note

The patient assessment workflow and the chat interface share the same RAG retrieval and reasoning logic directly, rather than exposing it through a separate shared API endpoint. This keeps the core behavior easy to inspect and debug. In a production setting, this would likely be factored into a dedicated internal service.

## Prompt Files

The system uses two prompt files located in `app/prompts/`:

- **`app/prompts/assessment.py`** — Part 1: Clinical Decision Support prompts. Contains the system and user prompts used by the assessment workflow (`/assess/{patient_id}`) to instruct Gemini on how to evaluate patient data against NG12 guideline passages and return structured JSON results.

- **`app/prompts/chat.py`** — Part 2: Conversational Chat prompts. Contains the system and user prompts for the chat workflow (`/chat`), including query rewriting, qualification, refusal templates, and citation formatting helpers.

## ChromaDB Collections

| Collection | Contents | Purpose | Query Method |
|------------|----------|---------|--------------|
| `ng12_canonical` | rule_canonical | Verbatim PDF text for citation display | ID-based lookup |
| `ng12_guidelines` | rule_search | Template-enriched text with synonym expansion for better vector retrieval | Vector similarity |
| `ng12_guidelines` | symptom_index | Symptom-to-cancer mapping with cross-references to Part A rules | Vector similarity |

## Notes

Assumptions, trade-offs, and future improvements.

### PDF Ingestion & Chunking

- **Chunk boundaries** — PDF parsing and chunk boundaries are not always perfectly aligned with clinical structure (e.g. criteria vs rationale vs action). Table extraction from the symptom index can be lossy, especially for age and duration thresholds.
- **Improvement** — With more time, ingestion would be made more structure-aware and metadata-driven.

### RAG Retrieval Limitations

- **Flat retrieval** — Current retrieval does not follow cross-references between chunks (e.g. "see section on...").
- **Soft reference expansion** — Detect cross-reference phrases during indexing, store referenced chunk IDs in metadata, and automatically include them at retrieval time.
- **Deep cross-referencing** — GraphRAG, hybrid search (Elasticsearch), and multi-step agentic RAG to validate thresholds and exceptions.

### Embedding Gap

- **Lay language vs clinical terminology** — User queries may use non-clinical language that does not match guideline terminology, reducing retrieval accuracy.

### Memory & Multi-turn Coherence

- **Multi-turn coherence** — Topic tracking, follow-up detection, and summary extraction across multi-turn conversations have room for improvement. Topic drift, history contamination, and ambiguous follow-ups remain challenging.
- **Improvements:**
  - Global-level symptom-focused memory for better multi-turn reasoning
  - Per-user context and preferences persistent across sessions
  - Topic-aware reranking (use session topic to rerank, not just prepend to queries)
  - Structured clinical state object (age, sex, symptoms, durations) updated incrementally per turn

### Guardrails

- **Input guardrail** — Uses deterministic regex/keyword matching (no LLM). Fast and predictable, but cannot catch nuanced edge cases.
- **Output guardrail** — Currently limited to citation validation and qualified response tiers. A dedicated output guardrail pass would better catch hallucinated recommendations and unsupported clinical claims.

### Guideline Document Control

- **Source & version tracking** — Track document source, version history, and update status to ensure the system always reflects the latest published guidelines.
