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

## Design Note

The patient assessment workflow and the chat interface share the same RAG retrieval and reasoning logic directly, rather than exposing it through a separate shared API endpoint. This keeps the core behavior easy to inspect and debug. In a production setting, this would likely be factored into a dedicated internal service.

## Prompt Files

The system uses two prompt files located in `app/prompts/`:

- **`app/prompts/assessment.py`** — Part 1: Clinical Decision Support prompts. Contains the system and user prompts used by the assessment workflow (`/assess/{patient_id}`) to instruct Gemini on how to evaluate patient data against NG12 guideline passages and return structured JSON results.

- **`app/prompts/chat.py`** — Part 2: Conversational Chat prompts. Contains the system and user prompts for the chat workflow (`/chat`), including query rewriting, qualification, refusal templates, and citation formatting helpers.
