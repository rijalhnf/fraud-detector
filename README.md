# AI Fraud Early Warning System Backend

FastAPI backend for a two-step fraud analysis workflow:

1. Upload PDF and extract raw financial variables for user validation.
2. Analyze validated variables with Beneish M-Score + RAG + local LLM insight.

## Features

- FastAPI REST API with CORS support.
- PDF text extraction via pdfplumber, with PyMuPDF fallback.
- Heuristic Beneish variable extraction with fallback defaults.
- Beneish 8-ratio calculation and final M-Score.
- RAG retrieval from ChromaDB documents (PSAK 72 and OJK-related references).
- Local LLM narrative generation via Ollama (`qwen3.5:9b`).
- Dockerized deployment with a production profile.

## Tech Stack and Versions

- Python: `3.12` (Docker base image: `python:3.12-slim`)
- API Framework: `FastAPI 0.116.1`
- ASGI Server: `Uvicorn 0.35.0` (`uvicorn[standard]`)
- HTTP Client: `requests 2.32.4`
- File Upload Parsing: `python-multipart 0.0.20`
- PDF Extraction: `pdfplumber 0.11.7`, `PyMuPDF 1.26.3`
- Vector Database: `ChromaDB 1.0.17`
- LLM Runtime: `Ollama` (model: `qwen3.5:9b`)
- Container Runtime: `Docker` + `Docker Compose`

## API Flow

### Endpoint 1: POST /api/upload

- Accepts a PDF file upload.
- Extracts text from PDF.
- Parses Beneish input variables heuristically.
- Returns extracted variables to frontend for user validation.

### Endpoint 2: POST /api/analyze

Accepts validated variables from frontend, then executes:

- Step A: Compute Beneish ratios and final M-Score.
- Step B: Query ChromaDB for relevant context.
- Step C: Build prompt and call Ollama.
- Step D: Return ratios, M-Score, risk status, and LLM narrative.

## Required Environment Variables

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `qwen3.5:9b`)
- `CHROMA_DB_PATH` (default: `./chroma_db`)
- `CHROMA_COLLECTION_NAME` (default: `fraud_knowledge`)

## Docker Run

### Standard service

```bash
docker compose build
docker compose up -d api
```

### Production profile (resource limits + healthcheck)

```bash
docker compose --profile production up -d api-prod
```

The `api-prod` service includes:

- Healthcheck to `/health`
- CPU limit: `2.0`
- Memory limit: `4g`
- Memory reservation: `1g`

### Stop services

```bash
docker compose down
```

## ChromaDB Data Persistence

- Host path: `./chroma_db`
- Container path: `/data/chroma`

Data remains on host volume between container restarts.

## How to Provide PSAK 72 and OJK Letter Documents

1. Put your source files in the `knowledge` folder:
  - Supported formats: `.pdf`, `.txt`, `.md`
  - Example names:
    - `knowledge/PSAK_72.pdf`
    - `knowledge/OJK_Sanction_Letter_2024.pdf`

2. Start the API container (if not running):

```bash
docker compose up -d api
```

3. Ingest documents into ChromaDB:

```bash
docker compose exec api python scripts/ingest_knowledge.py --input-dir /app/knowledge --db-path /data/chroma --collection fraud_knowledge --reset
```

4. Verify by calling `POST /api/analyze`.
  The endpoint will query the same collection configured by `CHROMA_COLLECTION_NAME`.

Notes:

- `--reset` clears old collection data first. Remove `--reset` to append/update.
- The folder is mounted in Docker via `./knowledge:/app/knowledge:ro`.

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
