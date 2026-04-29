# AI Fraud Early Warning System Backend

FastAPI backend for a two-step fraud analysis workflow:

1. Upload PDF and extract raw financial variables for user validation.
2. Analyze validated variables with Beneish M-Score + RAG + local LLM insight.

## Features

- FastAPI REST API with CORS support.
- AI vision OCR for short financial statement PDFs using the configured OpenRouter or Ollama model.
- PDF pages are rendered with PyMuPDF and sent to the model for structured variable extraction.
- Beneish 8-ratio calculation and final M-Score.
- RAG retrieval from ChromaDB documents (PSAK 115 / PSAK 72).
- Local LLM narrative generation via Ollama or OpenRouter.
- Dockerized deployment with a production profile.

## Tech Stack and Versions

- Python: `3.12` (Docker base image: `python:3.12-slim`)
- API Framework: `FastAPI 0.116.1`
- ASGI Server: `Uvicorn 0.35.0` (`uvicorn[standard]`)
- HTTP Client: `requests 2.32.4`
- File Upload Parsing: `python-multipart 0.0.20`
- PDF Rendering: `PyMuPDF 1.26.3`
- Vector Database: `ChromaDB 1.0.17`
- LLM Runtime: `OpenRouter` or `Ollama`
- Container Runtime: `Docker` + `Docker Compose`

## API Flow

### Endpoint 1: POST /api/upload

- Accepts a PDF file upload.
- Renders the PDF pages as images and sends them to the configured AI provider for OCR.
- Extracts Beneish input variables from the model response.
- Intended for short financial statement PDFs, usually about 2 to 4 pages.
- Returns extracted variables to frontend for user validation.

### Endpoint 2: POST /api/analyze

Accepts validated variables from frontend, then executes:

- Step A: Compute Beneish ratios and final M-Score.
- Step B: Query ChromaDB for relevant context.
- Step C: Build prompt and call Ollama.
- Step D: Return ratios, M-Score, risk status, and LLM narrative.

## Required Environment Variables

You must configure the AI provider using a `.env` file in the root directory:

```dotenv
AI_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemma-4-26b-a4b-it:free
OLLAMA_MODEL=gemma4:26b
```
*or AI_PROVIDER=local

Other internal variables set in `docker-compose.yml`:
- `OLLAMA_BASE_URL` (default: `http://host.docker.internal:11434`)
- `CHROMA_DB_PATH` (default: `/data/chroma`)
- `CHROMA_COLLECTION_NAME` (default: `fraud_knowledge`)

## Docker Run

### Quick Start (Step-by-Step)

**Step 1: Configure Environment Variables**
Ensure your `.env` file is set up as shown above.

**Step 2: Add Knowledge Base Documents (RAG)**
Put your source files (PSAK 115, OJK rules) in the `knowledge/` folder. Supported formats: `.pdf`, `.txt`, `.md`.

**Step 3: Build and Run the Docker Container**
Start the backend server in the background:
```bash
docker compose up --build -d
```

**Step 4: Ingest Knowledge Documents into ChromaDB**
Once the container is running, execute this command to parse and insert the PDFs into the vector database. You only need to run this once, or whenever you add new files to the `knowledge/` folder.
```bash
docker compose exec api python scripts/ingest_knowledge.py --input-dir /app/knowledge --db-path /data/chroma --collection fraud_knowledge --reset
```
When it finishes, the terminal will output the number of chunks added to the database.

**Step 5: Test the API**
Verify the backend is running by opening `http://localhost:8000/health` in your browser.

### Stop services

```bash
docker compose down
```

### Production profile (resource limits + healthcheck)

```bash
docker compose --profile production up -d api-prod
```

## ChromaDB Data Persistence

- Host path: `./chroma_db`
- Container path: `/data/chroma`

Data remains on host volume between container restarts.

## Ollama Integration Notes

In Docker Compose, `OLLAMA_BASE_URL` is set to:

- `http://host.docker.internal:11434`

Recommended for VPS security:

- Keep Ollama bound to localhost only (`127.0.0.1:11434`) and do not expose port `11434` publicly.
- Keep API public via port `8000` (or reverse proxy), while Ollama remains private.
- If Ollama runs on another private host, update this value in `docker-compose.yml`.

## Example Analyze Payload

Use this as JSON body for `POST /api/analyze`:

```json
{
  "receivables_t": 125000.0,
  "receivables_t1": 98000.0,
  "sales_t": 1200000.0,
  "sales_t1": 1050000.0,
  "cogs_t": 690000.0,
  "cogs_t1": 620000.0,
  "current_assets_t": 540000.0,
  "current_assets_t1": 500000.0,
  "ppe_t": 430000.0,
  "ppe_t1": 410000.0,
  "total_assets_t": 1450000.0,
  "total_assets_t1": 1360000.0,
  "depreciation_t": 47000.0,
  "depreciation_t1": 45000.0,
  "sga_expense_t": 175000.0,
  "sga_expense_t1": 162000.0,
  "current_liabilities_t": 320000.0,
  "current_liabilities_t1": 295000.0,
  "long_term_debt_t": 280000.0,
  "long_term_debt_t1": 260000.0,
  "income_from_operations_t": 133000.0,
  "cash_flow_from_operations_t": 101000.0,
  "company_name": "PT Contoh Nusantara",
  "fiscal_year": "2025"
}
```

## Project Files

- `main.py`: FastAPI app and fraud analysis flow.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container image build instructions.
- `docker-compose.yml`: Standard and production profile services.
- `.dockerignore`: Docker build context exclusions.
