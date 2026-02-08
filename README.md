# NG12 Cancer Risk Assessor

A clinical decision support system that uses RAG to assess cancer risk based on NICE NG12 guidelines.

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

### 3. Build the Vector Database

On first run, you need to index the NG12 PDF into ChromaDB. Go to the **Vector DB Admin** tab on the main page and click the **"Re-index PDF"** button. This will parse the PDF and populate the vector store — it only needs to be done once.

## Prompt Files

The system uses two prompt files located in `app/prompts/`:

- **`app/prompts/assessment.py`** — Part 1: Clinical Decision Support prompts. Contains the system and user prompts used by the assessment workflow (`/assess/{patient_id}`) to instruct Gemini on how to evaluate patient data against NG12 guideline passages and return structured JSON results.

- **`app/prompts/chat.py`** — Part 2: Conversational Chat prompts. Contains the system and user prompts for the chat workflow (`/chat`), including query rewriting, qualification, refusal templates, and citation formatting helpers.
