from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

load_dotenv()

# --- CORS Settings ---
# Allow requests from these frontend origins (for browser apps)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://rij.al",
    "https://www.rij.al",
]

# --- AI Configuration ---
# Read from environment variables (populated by .env or Docker)
AI_PROVIDER = os.getenv("AI_PROVIDER")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "4"))
OCR_RENDER_SCALE = float(os.getenv("OCR_RENDER_SCALE", "2.0"))

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
    extracted_variables: dict[str, float | None]
    ocr_provider: str | None = None
    ocr_model: str | None = None
    ocr_raw_response: str | None = None
    responded_at: str
    duration_ms: float


class FinancialVariables(BaseModel):
    """
    User-validated financial variables required for Beneish M-Score calculation.
    Use _t for current year and _t1 for prior year.
    Beneish M-Score requires prior-year values for the core ratio inputs.
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
                "llm_provider": "openrouter",
                "llm_model": "google/gemma-4-26b-a4b-it",
                "llm_raw_response": "{\"id\":\"...\",\"choices\":[...]}",
                "responded_at": "2026-04-26T12:00:00.000000Z",
                "duration_ms": 823.57,
            }
        }
    )

    ratios: dict[str, float]
    m_score: float
    risk_status: str
    llm_narrative_insight: str
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_raw_response: str | None = None
    responded_at: str
    duration_ms: float


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_div(numerator: float | None, denominator: float | None, label: str) -> float:
    if numerator is None or denominator is None:
        return 1.0  # Return neutral 1.0 ratio if missing previous year data
    if denominator == 0:
        return 1.0  # Avoid division by zero by returning neutral ratio
    return numerator / denominator


def calculate_beneish_ratios(data: FinancialVariables) -> dict[str, float]:
    sales_t1 = data.sales_t1
    cogs_t1 = data.cogs_t1
    receivables_t1 = data.receivables_t1
    current_assets_t1 = data.current_assets_t1
    ppe_t1 = data.ppe_t1
    total_assets_t1 = data.total_assets_t1
    depreciation_t1 = data.depreciation_t1
    sga_expense_t1 = data.sga_expense_t1
    current_liabilities_t1 = data.current_liabilities_t1
    long_term_debt_t1 = data.long_term_debt_t1

    dsri = _safe_div(
        _safe_div(data.receivables_t, data.sales_t, "DSRI receivables_t/sales_t"),
        _safe_div(receivables_t1, sales_t1, "DSRI receivables_t1/sales_t1"),
        "DSRI",
    )

    gross_margin_t = _safe_div(data.sales_t - data.cogs_t, data.sales_t, "GMI gross_margin_t")
    gross_margin_t1 = _safe_div(
        sales_t1 - cogs_t1, sales_t1, "GMI gross_margin_t1"
    )
    gmi = _safe_div(gross_margin_t1, gross_margin_t, "GMI")

    asset_quality_t = 1 - _safe_div(
        data.current_assets_t + data.ppe_t,
        data.total_assets_t,
        "AQI asset_quality_t",
    )
    asset_quality_t1 = 1 - _safe_div(
        current_assets_t1 + ppe_t1,
        total_assets_t1,
        "AQI asset_quality_t1",
    )
    aqi = _safe_div(asset_quality_t, asset_quality_t1, "AQI")

    sgi = _safe_div(data.sales_t, sales_t1, "SGI")

    depi = _safe_div(
        _safe_div(
            depreciation_t1,
            depreciation_t1 + ppe_t1,
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
        _safe_div(sga_expense_t1, sales_t1, "SGAI sga_expense_t1/sales_t1"),
        "SGAI",
    )

    lvgi = _safe_div(
        _safe_div(
            data.current_liabilities_t + data.long_term_debt_t,
            data.total_assets_t,
            "LVGI leverage_t",
        ),
        _safe_div(
            current_liabilities_t1 + long_term_debt_t1,
            total_assets_t1,
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


def _default_mock_variables() -> dict[str, float | None]:
    return {
        "receivables_t": None,
        "receivables_t1": None,
        "sales_t": None,
        "sales_t1": None,
        "cogs_t": None,
        "cogs_t1": None,
        "current_assets_t": None,
        "current_assets_t1": None,
        "ppe_t": None,
        "ppe_t1": None,
        "total_assets_t": None,
        "total_assets_t1": None,
        "depreciation_t": None,
        "depreciation_t1": None,
        "sga_expense_t": None,
        "sga_expense_t1": None,
        "current_liabilities_t": None,
        "current_liabilities_t1": None,
        "long_term_debt_t": None,
        "long_term_debt_t1": None,
        "income_from_operations_t": None,
        "cash_flow_from_operations_t": None,
    }


def _parse_raw_value(raw_value: str) -> float | None:
    raw_value = raw_value.replace(" ", "")
    # Handle European format (1.000.000,00 vs 1,000,000.00)
    if "," in raw_value and "." in raw_value:
        raw_value = raw_value.replace(",", "")
    else:
        raw_value = raw_value.replace(",", ".")
    try:
        return float(raw_value)
    except ValueError:
        return None


def _render_pdf_pages_for_ocr(pdf_bytes: bytes, max_pages: int = OCR_MAX_PAGES) -> tuple[list[str], int]:
    try:
        import fitz  # type: ignore
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"PyMuPDF is required for OCR rendering: {exc}") from exc

    images: list[str] = []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            pages_to_render = min(page_count, max_pages)
            matrix = fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE)

            for page_index in range(pages_to_render):
                page = doc.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(base64.b64encode(pixmap.tobytes("png")).decode("ascii"))

            return images, page_count
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF pages for AI OCR: {exc}") from exc


def _build_ocr_prompt(page_count: int, processed_pages: int) -> str:
    fields = list(_default_mock_variables().keys())
    fields_json = json.dumps({field: None for field in fields}, ensure_ascii=True, indent=2)

    return (
        "You are an OCR and extraction engine for financial statements. "
        "Read the supplied PDF page images and extract only the numeric values requested below.\n\n"
        f"The PDF has {page_count} page(s). Process only the first {processed_pages} page(s). "
        "These statements are short, usually 2-4 pages, and contain balance sheet and income statement data.\n\n"
        "Return valid JSON only. Do not wrap it in markdown. Do not add explanations.\n"
        "Rules:\n"
        "- Use the exact keys shown below.\n"
        "- Return numbers as plain JSON numbers when possible.\n"
        "- Use null if a value is unreadable or not present.\n"
        "- Do not guess values.\n"
        "- If the statement uses comma or dot separators, normalize them into numeric JSON values.\n\n"
        f"Keys and default shape:\n{fields_json}\n"
    )


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    candidates: list[str] = []
    stripped = raw_text.strip()
    if stripped:
        candidates.append(stripped)

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    start_index = raw_text.find("{")
    end_index = raw_text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        candidates.append(raw_text[start_index : end_index + 1].strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    raise ValueError("AI OCR response did not contain a valid JSON object.")


def _normalize_ocr_value(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _parse_raw_value(value)
    return None


def _normalize_ocr_variables(payload: Any) -> tuple[dict[str, float | None], int]:
    raw_variables = payload.get("extracted_variables") if isinstance(payload, dict) else payload
    if not isinstance(raw_variables, dict):
        raw_variables = payload if isinstance(payload, dict) else {}

    normalized = _default_mock_variables()
    extracted_count = 0

    for key in normalized.keys():
        value = _normalize_ocr_value(raw_variables.get(key))
        normalized[key] = value
        if value is not None:
            extracted_count += 1

    return normalized, extracted_count


def _call_openrouter_vision(prompt: str, images_b64: list[str]) -> str:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is missing.")
    if not OPENROUTER_MODEL:
        raise HTTPException(status_code=500, detail="OPENROUTER_MODEL is missing.")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_b64 in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            }
        )

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise OCR extraction assistant."},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_ollama_vision(prompt: str, images_b64: list[str]) -> str:
    if not OLLAMA_MODEL:
        raise HTTPException(status_code=500, detail="OLLAMA_MODEL is missing.")

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192,
        },
    }

    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


def extract_financial_variables(pdf_bytes: bytes) -> tuple[dict[str, float | None], str, str, str, str]:
    """
    Uses the selected AI provider to OCR short financial statement PDFs and extract Beneish inputs.
    """
    images_b64, page_count = _render_pdf_pages_for_ocr(pdf_bytes)
    processed_pages = len(images_b64)

    if not images_b64:
        raise HTTPException(status_code=400, detail="Uploaded PDF does not contain any renderable pages.")

    prompt = _build_ocr_prompt(page_count=page_count, processed_pages=processed_pages)

    provider = (AI_PROVIDER or "").strip().lower()
    if provider == "openrouter":
        raw_response = _call_openrouter_vision(prompt, images_b64)
    elif provider == "ollama":
        raw_response = _call_ollama_vision(prompt, images_b64)
    else:
        raise HTTPException(
            status_code=500,
            detail="AI_PROVIDER must be set to either 'openrouter' or 'ollama' for OCR.",
        )

    try:
        extracted_payload = _extract_json_object(raw_response)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=f"AI OCR response could not be parsed: {exc}") from exc

    parsed_variables, extracted_count = _normalize_ocr_variables(extracted_payload)
    if extracted_count == 0 and isinstance(extracted_payload, dict):
        parsed_variables = _default_mock_variables()
        for key, value in extracted_payload.items():
            if key in parsed_variables:
                parsed_variables[key] = _normalize_ocr_value(value)
        extracted_count = sum(1 for value in parsed_variables.values() if value is not None)

    note = (
        f"AI OCR extraction via {provider} completed on {processed_pages}/{page_count} page(s). "
        f"Extracted {extracted_count}/{len(parsed_variables)} variables."
    )
    model_name = OPENROUTER_MODEL if provider == "openrouter" else OLLAMA_MODEL
    return parsed_variables, note, provider, model_name or "", raw_response


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
                        "PSAK 115 generally addresses revenue recognition rules. "
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
    benchmark_reference = {
        "DSRI": {
            "non_manipulator_mean": 1.031,
            "red_flag_threshold": 1.465,
            "insight": "Indikasi channel stuffing atau pengakuan pendapatan terlalu dini.",
        },
        "GMI": {
            "non_manipulator_mean": 1.014,
            "red_flag_threshold": 1.193,
            "insight": "Margin memburuk dapat mendorong manajemen menyamarkan rugi.",
        },
        "AQI": {
            "non_manipulator_mean": 1.039,
            "red_flag_threshold": 1.254,
            "insight": "Kualitas aset menurun karena deferral atau kapitalisasi biaya agresif.",
        },
        "SGI": {
            "non_manipulator_mean": 1.134,
            "red_flag_threshold": 1.607,
            "insight": "Pertumbuhan tinggi meningkatkan tekanan untuk mempertahankan target.",
        },
        "DEPI": {
            "non_manipulator_mean": 1.001,
            "red_flag_threshold": 1.077,
            "insight": "Perlambatan penyusutan dapat menaikkan laba secara artifisial.",
        },
        "SGAI": {
            "non_manipulator_mean": 1.054,
            "red_flag_threshold": 1.041,
            "insight": "Efisiensi beban SG&A melemah terhadap pendapatan.",
        },
        "LVGI": {
            "non_manipulator_mean": 1.037,
            "red_flag_threshold": 1.111,
            "insight": "Leverage naik meningkatkan risiko manipulasi demi covenant utang.",
        },
        "TATA": {
            "non_manipulator_mean": 0.018,
            "red_flag_threshold": 0.031,
            "insight": "Akrual tinggi dibanding arus kas menandakan kualitas laba rendah.",
        },
    }

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

Retrieved Context (RAG from PSAK 115 and OJK sanction-related references):
{chr(10).join(context_lines)}

Beneish Benchmark Reference (for metric-level red flags):
{json.dumps(benchmark_reference, ensure_ascii=True, indent=2)}

Instructions:
1) Explain what the M-Score means for this company in plain but professional language.
2) Connect findings to the retrieved context where relevant.
3) Bandingkan setiap rasio Beneish terhadap benchmark referensi dan sorot metrik red flag.
4) Provide concise red flags and recommended follow-up checks.
5) Keep response under 220 words.
6) Write the entire response in Bahasa Indonesia (not English).
""".strip()


def call_llm(prompt: str) -> tuple[str, str, str, str]:
    """Routes the prompt to either OpenRouter or local Ollama based on AI_PROVIDER constant."""
    provider = (AI_PROVIDER or "").lower()
    if provider == "openrouter":
        narrative, model_provider, model_name, raw_response = _call_openrouter(prompt)
    else:
        narrative, model_provider, model_name, raw_response = _call_ollama(prompt)

    if _needs_indonesian_rewrite(narrative):
        rewritten_text, rewrite_raw = _rewrite_to_bahasa_indonesia(narrative, provider)
        if rewritten_text:
            narrative = rewritten_text
            raw_response = json.dumps(
                {
                    "initial_raw_response": raw_response,
                    "rewrite_raw_response": rewrite_raw,
                },
                ensure_ascii=True,
            )

    return narrative, model_provider, model_name, raw_response


def _needs_indonesian_rewrite(text: str) -> bool:
    normalized = f" {text.lower()} "
    english_markers = [
        " the ",
        " and ",
        " with ",
        " company ",
        " revenue ",
        " risk ",
        " recommend ",
        " therefore ",
    ]
    indonesian_markers = [
        " dan ",
        " yang ",
        " dengan ",
        " terhadap ",
        " perusahaan ",
        " pendapatan ",
        " risiko ",
        " rekomendasi ",
    ]

    english_hits = sum(1 for marker in english_markers if marker in normalized)
    indonesian_hits = sum(1 for marker in indonesian_markers if marker in normalized)

    return english_hits > indonesian_hits


def _rewrite_to_bahasa_indonesia(text: str, provider: str) -> tuple[str | None, str]:
    rewrite_prompt = (
        "Tulis ulang teks berikut seluruhnya dalam Bahasa Indonesia profesional. "
        "Jangan ubah fakta, angka, atau kesimpulan. Hanya ubah bahasanya.\n\n"
        f"Teks:\n{text}"
    )

    try:
        if provider == "openrouter":
            rewritten, _, _, raw = _call_openrouter(rewrite_prompt)
            return rewritten, raw
        rewritten, _, _, raw = _call_ollama(rewrite_prompt)
        return rewritten, raw
    except Exception as exc:
        return None, json.dumps({"rewrite_error": str(exc)}, ensure_ascii=True)


def _call_ollama(prompt: str) -> tuple[str, str, str, str]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    payload = {
        "model": OLLAMA_MODEL,
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
        return (
            data.get("response", "No response returned by Ollama.").strip(),
            "ollama",
            OLLAMA_MODEL or "",
            json.dumps(data, ensure_ascii=True),
        )
    except Exception as exc:
        fallback = (
            "Layanan LLM tidak tersedia. Menggunakan insight deterministik: "
            f"M-Score menunjukkan '{'risiko meningkat' if 'High Risk' in prompt else 'risiko lebih rendah'}'. "
            f"Detail error: {exc}"
        )
        return fallback, "ollama", OLLAMA_MODEL or "", json.dumps({"error": str(exc)}, ensure_ascii=True)


def _call_openrouter(prompt: str) -> tuple[str, str, str, str]:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        message = "OpenRouter API key belum diatur. Silakan set OPENROUTER_API_KEY di environment."
        return message, "openrouter", OPENROUTER_MODEL or "", json.dumps({"error": message}, ensure_ascii=True)
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "Selalu jawab dalam Bahasa Indonesia profesional."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return (
            data["choices"][0]["message"]["content"].strip(),
            "openrouter",
            OPENROUTER_MODEL or "",
            json.dumps(data, ensure_ascii=True),
        )
    except Exception as exc:
        fallback = (
            "Layanan LLM tidak tersedia. Menggunakan insight deterministik: "
            f"M-Score menunjukkan '{'risiko meningkat' if 'High Risk' in prompt else 'risiko lebih rendah'}'. "
            f"Detail error: {exc}"
        )
        return fallback, "openrouter", OPENROUTER_MODEL or "", json.dumps({"error": str(exc)}, ensure_ascii=True)


@app.get("/health")
def health() -> dict[str, str | float]:
    started_at = time.perf_counter()
    return {
        "status": "ok",
        "responded_at": _iso_utc_now(),
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    started_at = time.perf_counter()

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    extracted_variables, extraction_note, ocr_provider, ocr_model, ocr_raw_response = extract_financial_variables(pdf_bytes)
    preview = (
        f"AI OCR processed {file.filename or 'uploaded.pdf'} and extracted "
        f"{sum(1 for value in extracted_variables.values() if value is not None)} financial fields."
    )

    return UploadResponse(
        filename=file.filename or "uploaded.pdf",
        extraction_note=extraction_note,
        extracted_text_preview=preview,
        extracted_variables=extracted_variables,
        ocr_provider=ocr_provider,
        ocr_model=ocr_model,
        ocr_raw_response=ocr_raw_response,
        responded_at=_iso_utc_now(),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
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
    started_at = time.perf_counter()

    # Step A (Math): Beneish ratios + final M-Score.
    ratios = calculate_beneish_ratios(payload)
    m_score = calculate_m_score(ratios)
    risk_status = classify_risk(m_score)

    # Step B (RAG): Query ChromaDB for context from PSAK 115 and OJK-related docs.
    rag_query = (
        f"Beneish M-Score {m_score:.4f}, risk {risk_status}, "
        "find context from PSAK 115 and OJK sanction letter references"
    )
    rag_result = query_chromadb_context(rag_query)
    rag_chunks = rag_result.get("chunks", [])

    # Step C (LLM): Combine math output + RAG context and send to the selected AI provider.
    prompt = build_llm_prompt(payload, ratios, m_score, risk_status, rag_chunks)
    llm_narrative, llm_provider, llm_model, llm_raw_response = call_llm(prompt)

    # Step D (Response): Return complete fraud signal output.
    return AnalyzeResponse(
        ratios={k: round(v, 6) for k, v in ratios.items()},
        m_score=round(m_score, 6),
        risk_status=risk_status,
        llm_narrative_insight=llm_narrative,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_raw_response=llm_raw_response,
        responded_at=_iso_utc_now(),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
