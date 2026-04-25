from __future__ import annotations

import io
import json
import os
import re
from typing import Any

import requests
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


# --- CORS Settings ---
# Allow requests from these frontend origins (for browser apps)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://rij.al",
    "https://www.rij.al",
]

app = FastAPI(
    title="AI Fraud Early Warning System API",
    version="1.0.0",
    description="Two-step fraud analysis flow: Upload -> Analyze",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    filename: str
    extraction_note: str
    extracted_text_preview: str
    extracted_variables: dict[str, float]


class FinancialVariables(BaseModel):
    """
    User-validated financial variables required for Beneish M-Score calculation.
    Use _t for current year and _t1 for prior year.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
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
                "fiscal_year": "2025",
            }
        },
    )

    receivables_t: float = Field(..., description="Net receivables for current year")
    receivables_t1: float = Field(..., description="Net receivables for prior year")

    sales_t: float = Field(..., description="Net sales/revenue for current year")
    sales_t1: float = Field(..., description="Net sales/revenue for prior year")

    cogs_t: float = Field(..., description="Cost of goods sold for current year")
    cogs_t1: float = Field(..., description="Cost of goods sold for prior year")

    current_assets_t: float = Field(..., description="Current assets for current year")
    current_assets_t1: float = Field(..., description="Current assets for prior year")

    ppe_t: float = Field(..., description="Net property, plant and equipment current year")
    ppe_t1: float = Field(..., description="Net property, plant and equipment prior year")

    total_assets_t: float = Field(..., description="Total assets for current year")
    total_assets_t1: float = Field(..., description="Total assets for prior year")

    depreciation_t: float = Field(..., description="Depreciation expense current year")
    depreciation_t1: float = Field(..., description="Depreciation expense prior year")

    sga_expense_t: float = Field(..., description="SG&A expense current year")
    sga_expense_t1: float = Field(..., description="SG&A expense prior year")

    current_liabilities_t: float = Field(..., description="Current liabilities current year")
    current_liabilities_t1: float = Field(..., description="Current liabilities prior year")

    long_term_debt_t: float = Field(..., description="Long-term debt current year")
    long_term_debt_t1: float = Field(..., description="Long-term debt prior year")

    income_from_operations_t: float = Field(
        ..., description="Income from operations current year"
    )
    cash_flow_from_operations_t: float = Field(
        ..., description="Cash flow from operations current year"
    )

    company_name: str | None = Field(default=None)
    fiscal_year: str | None = Field(default=None)


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ratios": {
                    "DSRI": 1.115646,
                    "GMI": 1.029057,
                    "AQI": 0.964824,
                    "SGI": 1.142857,
                    "DEPI": 1.004246,
                    "SGAI": 1.016129,
                    "LVGI": 1.048997,
                    "TATA": 0.022069,
                },
                "m_score": -2.231455,
                "risk_status": "Medium Risk (Watchlist)",
                "llm_narrative_insight": "Revenue growth and accrual quality warrant focused review.",
            }
        }
    )

    ratios: dict[str, float]
    m_score: float
    risk_status: str
    llm_narrative_insight: str


def _safe_div(numerator: float, denominator: float, label: str) -> float:
    if denominator == 0:
        raise HTTPException(status_code=422, detail=f"Division by zero in {label}")
    return numerator / denominator


def calculate_beneish_ratios(data: FinancialVariables) -> dict[str, float]:
    dsri = _safe_div(
        _safe_div(data.receivables_t, data.sales_t, "DSRI receivables_t/sales_t"),
        _safe_div(data.receivables_t1, data.sales_t1, "DSRI receivables_t1/sales_t1"),
        "DSRI",
    )

    gross_margin_t = _safe_div(data.sales_t - data.cogs_t, data.sales_t, "GMI gross_margin_t")
    gross_margin_t1 = _safe_div(
        data.sales_t1 - data.cogs_t1, data.sales_t1, "GMI gross_margin_t1"
    )
    gmi = _safe_div(gross_margin_t1, gross_margin_t, "GMI")

    asset_quality_t = 1 - _safe_div(
        data.current_assets_t + data.ppe_t,
        data.total_assets_t,
        "AQI asset_quality_t",
    )
    asset_quality_t1 = 1 - _safe_div(
        data.current_assets_t1 + data.ppe_t1,
        data.total_assets_t1,
        "AQI asset_quality_t1",
    )
    aqi = _safe_div(asset_quality_t, asset_quality_t1, "AQI")

    sgi = _safe_div(data.sales_t, data.sales_t1, "SGI")

    depi = _safe_div(
        _safe_div(
            data.depreciation_t1,
            data.depreciation_t1 + data.ppe_t1,
            "DEPI depreciation_t1/(depreciation_t1+ppe_t1)",
        ),
        _safe_div(
            data.depreciation_t,
            data.depreciation_t + data.ppe_t,
            "DEPI depreciation_t/(depreciation_t+ppe_t)",
        ),
        "DEPI",
    )

    sgai = _safe_div(
        _safe_div(data.sga_expense_t, data.sales_t, "SGAI sga_expense_t/sales_t"),
        _safe_div(data.sga_expense_t1, data.sales_t1, "SGAI sga_expense_t1/sales_t1"),
        "SGAI",
    )

    lvgi = _safe_div(
        _safe_div(
            data.current_liabilities_t + data.long_term_debt_t,
            data.total_assets_t,
            "LVGI leverage_t",
        ),
        _safe_div(
            data.current_liabilities_t1 + data.long_term_debt_t1,
            data.total_assets_t1,
            "LVGI leverage_t1",
        ),
        "LVGI",
    )

    tata = _safe_div(
        data.income_from_operations_t - data.cash_flow_from_operations_t,
        data.total_assets_t,
        "TATA",
    )

    return {
        "DSRI": dsri,
        "GMI": gmi,
        "AQI": aqi,
        "SGI": sgi,
        "DEPI": depi,
        "SGAI": sgai,
        "LVGI": lvgi,
        "TATA": tata,
    }


def calculate_m_score(ratios: dict[str, float]) -> float:
    return (
        -4.84
        + 0.92 * ratios["DSRI"]
        + 0.528 * ratios["GMI"]
        + 0.404 * ratios["AQI"]
        + 0.892 * ratios["SGI"]
        + 0.115 * ratios["DEPI"]
        - 0.172 * ratios["SGAI"]
        + 4.679 * ratios["TATA"]
        - 0.327 * ratios["LVGI"]
    )


def classify_risk(m_score: float) -> str:
    if m_score > -2.22:
        return "High Risk (Likely Earnings Manipulator)"
    if m_score > -2.6:
        return "Medium Risk (Watchlist)"
    return "Low Risk"


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Tries pdfplumber first, then PyMuPDF. Returns empty string if extraction fails.
    """
    text_parts: list[str] = []

    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    try:
        import fitz  # type: ignore

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text_parts.append(page.get_text("text") or "")
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    return ""


def _default_mock_variables() -> dict[str, float]:
    return {
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
    }


def _extract_number_near_label(text: str, labels: list[str]) -> float | None:
    number_pattern = r"(-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?)"
    for label in labels:
        pattern = rf"{label}\s*[:=-]?\s*{number_pattern}"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue

        raw_value = match.group(1).replace(" ", "")
        if "," in raw_value and "." in raw_value:
            raw_value = raw_value.replace(",", "")
        else:
            raw_value = raw_value.replace(",", ".")
        try:
            return float(raw_value)
        except ValueError:
            continue
    return None


def extract_financial_variables(extracted_text: str) -> tuple[dict[str, float], str]:
    """
    Heuristic parser that maps common financial labels to Beneish inputs.
    Falls back to mocked values for any field not found in PDF text.
    """
    fallback = _default_mock_variables()
    lower_text = extracted_text.lower()

    label_map: dict[str, list[str]] = {
        "receivables_t": [
            r"receivables\s*\(?t\)?",
            r"net\s*receivables\s*\(?current\)?",
        ],
        "receivables_t1": [
            r"receivables\s*\(?t-?1\)?",
            r"net\s*receivables\s*\(?prior\)?",
        ],
        "sales_t": [r"sales\s*\(?t\)?", r"revenue\s*\(?current\)?"],
        "sales_t1": [r"sales\s*\(?t-?1\)?", r"revenue\s*\(?prior\)?"],
        "cogs_t": [r"cogs\s*\(?t\)?", r"cost\s*of\s*goods\s*sold\s*\(?current\)?"],
        "cogs_t1": [r"cogs\s*\(?t-?1\)?", r"cost\s*of\s*goods\s*sold\s*\(?prior\)?"],
        "current_assets_t": [r"current\s*assets\s*\(?t\)?"],
        "current_assets_t1": [r"current\s*assets\s*\(?t-?1\)?"],
        "ppe_t": [r"ppe\s*\(?t\)?", r"property[,\s]*plant[,\s]*equipment\s*\(?current\)?"],
        "ppe_t1": [r"ppe\s*\(?t-?1\)?", r"property[,\s]*plant[,\s]*equipment\s*\(?prior\)?"],
        "total_assets_t": [r"total\s*assets\s*\(?t\)?"],
        "total_assets_t1": [r"total\s*assets\s*\(?t-?1\)?"],
        "depreciation_t": [r"depreciation\s*\(?t\)?", r"depreciation\s*expense\s*\(?current\)?"],
        "depreciation_t1": [r"depreciation\s*\(?t-?1\)?", r"depreciation\s*expense\s*\(?prior\)?"],
        "sga_expense_t": [r"sg&a\s*\(?t\)?", r"selling[,\s]*general[,\s]*admin\s*\(?current\)?"],
        "sga_expense_t1": [r"sg&a\s*\(?t-?1\)?", r"selling[,\s]*general[,\s]*admin\s*\(?prior\)?"],
        "current_liabilities_t": [r"current\s*liabilities\s*\(?t\)?"],
        "current_liabilities_t1": [r"current\s*liabilities\s*\(?t-?1\)?"],
        "long_term_debt_t": [r"long\s*term\s*debt\s*\(?t\)?"],
        "long_term_debt_t1": [r"long\s*term\s*debt\s*\(?t-?1\)?"],
        "income_from_operations_t": [
            r"income\s*from\s*operations\s*\(?t\)?",
            r"operating\s*income\s*\(?current\)?",
        ],
        "cash_flow_from_operations_t": [
            r"cash\s*flow\s*from\s*operations\s*\(?t\)?",
            r"operating\s*cash\s*flow\s*\(?current\)?",
        ],
    }

    parsed = fallback.copy()
    parsed_count = 0
    for key, labels in label_map.items():
        value = _extract_number_near_label(lower_text, labels)
        if value is not None:
            parsed[key] = value
            parsed_count += 1

    note = (
        "Heuristic extraction from PDF text with fallback defaults. "
        f"Parsed {parsed_count}/{len(label_map)} variables from detected labels."
    )
    return parsed, note


def query_chromadb_context(
    query_text: str,
    db_path: str | None = None,
    collection_name: str | None = None,
    top_k: int = 4,
) -> dict[str, Any]:
    db_path = db_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")
    collection_name = collection_name or os.getenv(
        "CHROMA_COLLECTION_NAME", "fraud_knowledge"
    )

    try:
        import chromadb  # type: ignore

        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(name=collection_name)
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks: list[dict[str, Any]] = []
        for i, doc in enumerate(docs):
            chunks.append(
                {
                    "content": doc,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )

        return {
            "source": "chromadb",
            "collection": collection_name,
            "chunks": chunks,
        }
    except Exception as exc:
        return {
            "source": "fallback",
            "collection": collection_name,
            "chunks": [
                {
                    "content": (
                        "PSAK 72 generally addresses revenue recognition rules. "
                        "OJK sanction letters may indicate governance, disclosure, or reporting concerns."
                    ),
                    "metadata": {"fallback_reason": str(exc)},
                    "distance": None,
                }
            ],
        }


def build_llm_prompt(
    data: FinancialVariables,
    ratios: dict[str, float],
    m_score: float,
    risk_status: str,
    rag_chunks: list[dict[str, Any]],
) -> str:
    context_lines = []
    for idx, chunk in enumerate(rag_chunks, start=1):
        context_lines.append(
            f"[{idx}] {chunk.get('content', '')}\nMetadata: {json.dumps(chunk.get('metadata', {}), ensure_ascii=True)}"
        )

    company = data.company_name or "Unknown Company"
    year = data.fiscal_year or "Unknown Fiscal Year"

    return f"""
You are a forensic accounting assistant for an AI Fraud Early Warning System.
Analyze the company's fraud risk using Beneish M-Score results and regulatory/accounting context.

Company: {company}
Fiscal Year: {year}

Calculated Beneish Ratios:
{json.dumps(ratios, indent=2)}

Final M-Score: {m_score:.4f}
Risk Status: {risk_status}

Retrieved Context (RAG from PSAK 72 and OJK sanction-related references):
{chr(10).join(context_lines)}

Instructions:
1) Explain what the M-Score means for this company in plain but professional language.
2) Connect findings to the retrieved context where relevant.
3) Provide concise red flags and recommended follow-up checks.
4) Keep response under 220 words.
""".strip()


def call_ollama(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
        },
    }

    try:
        response = requests.post(
            f"{base_url}/api/generate", json=payload, timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response returned by Ollama.").strip()
    except Exception as exc:
        return (
            "LLM service unavailable. Returning deterministic insight: "
            f"M-Score indicates '{'elevated risk' if 'High Risk' in prompt else 'lower risk'}'. "
            f"Error detail: {exc}"
        )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    extracted_text = extract_text_from_pdf(pdf_bytes)
    extracted_variables, extraction_note = extract_financial_variables(extracted_text)

    preview = extracted_text[:1500] if extracted_text else "No text extracted from PDF."

    return UploadResponse(
        filename=file.filename or "uploaded.pdf",
        extraction_note=extraction_note,
        extracted_text_preview=preview,
        extracted_variables=extracted_variables,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze_validated_data(
    payload: FinancialVariables = Body(
        ...,
        examples={
            "sample_payload": {
                "summary": "Validated variables from frontend review",
                "value": {
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
                    "fiscal_year": "2025",
                },
            }
        },
    )
) -> AnalyzeResponse:
    # Step A (Math): Beneish ratios + final M-Score.
    ratios = calculate_beneish_ratios(payload)
    m_score = calculate_m_score(ratios)
    risk_status = classify_risk(m_score)

    # Step B (RAG): Query ChromaDB for context from PSAK 72 and OJK-related docs.
    rag_query = (
        f"Beneish M-Score {m_score:.4f}, risk {risk_status}, "
        "find context from PSAK 72 and OJK sanction letter references"
    )
    rag_result = query_chromadb_context(rag_query)
    rag_chunks = rag_result.get("chunks", [])

    # Step C (LLM): Combine math output + RAG context and send to local Ollama.
    prompt = build_llm_prompt(payload, ratios, m_score, risk_status, rag_chunks)
    llm_narrative = call_ollama(prompt)

    # Step D (Response): Return complete fraud signal output.
    return AnalyzeResponse(
        ratios={k: round(v, 6) for k, v in ratios.items()},
        m_score=round(m_score, 6),
        risk_status=risk_status,
        llm_narrative_insight=llm_narrative,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
