from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
import httpx
import asyncio
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

llm_queue_lock = asyncio.Lock()

from prompts_analyze import build_llm_prompt, build_calk_prompt
from prompts_ocr import build_ocr_prompt
from models import UploadResponse, FinancialVariables, AnalyzeResponse, AnalyzeCalkResponse

load_dotenv()

# --- CORS Settings ---
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "https://rij.al",
    "https://www.rij.al",
]

# --- AI Configuration ---
AI_PROVIDER = os.getenv("AI_PROVIDER")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")
OPENROUTER_VISION_MODEL = os.getenv("OPENROUTER_VISION_MODEL", OPENROUTER_MODEL)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "10"))
OCR_RENDER_SCALE = float(os.getenv("OCR_RENDER_SCALE", "2.0"))

app = FastAPI(
    title="AI Fraud Early Warning System API",
    version="1.1.0",
    description="Two-step fraud analysis flow: Upload -> Analyze",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models have been moved to models.py


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _extract_openrouter_cost_usd(response_json: dict[str, Any]) -> float | None:
    usage = response_json.get("usage")
    candidates: list[Any] = []
    if isinstance(usage, dict):
        candidates.extend([
            usage.get("cost"),
            usage.get("total_cost"),
            usage.get("estimated_cost"),
            usage.get("estimated_cost_usd"),
        ])
    candidates.extend([
        response_json.get("cost"),
        response_json.get("total_cost"),
        response_json.get("estimated_cost"),
        response_json.get("estimated_cost_usd"),
    ])
    for candidate in candidates:
        parsed = _to_float_or_none(candidate)
        if parsed is not None:
            return parsed
    return None


def _safe_div(numerator: float | None, denominator: float | None, label: str) -> float:
    if numerator is None or denominator is None:
        return 1.0
    if denominator == 0:
        return 1.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Beneish M-Score calculation (methodology-corrected)
# ---------------------------------------------------------------------------

def calculate_beneish_ratios(data: FinancialVariables) -> dict[str, float]:
    """
    Compute the 8 Beneish (1999) ratios with corrected methodology:

    DSRI : Uses trade receivables only (receivables_t/t1), not other receivables.
    GMI  : Gross margin index — for airlines, gross margin = sales - total operating expenses.
    AQI  : Asset quality index — (1 - (CA + net PPE) / TA).
    SGI  : Sales growth index.
    DEPI : Depreciation index — uses actual depreciation from notes (not ΔAccum. dep).
    SGAI : SG&A index — uses full SG&A = sga_expense + selling_expense (Beneish original).
    LVGI : Leverage index — (current liabilities + LT interest-bearing debt) / total assets.
    TATA : Total accruals to total assets — income statement method: (Op. Income - CFO) / TA.
    """

    # ------------------------------------------------------------------
    # DSRI — Days Sales in Receivables Index
    # Detects premature revenue recognition or channel stuffing.
    # Rising DSRI means receivables growing faster than sales → red flag.
    # ------------------------------------------------------------------
    # If split fields are provided, override receivables_t with their sum
    rec_t = data.receivables_t
    rec_t1 = data.receivables_t1

    if (data.receivables_related_t + data.receivables_thirdparty_t) > 0:
        rec_t = data.receivables_related_t + data.receivables_thirdparty_t
    if (data.receivables_related_t1 + data.receivables_thirdparty_t1) > 0:
        rec_t1 = data.receivables_related_t1 + data.receivables_thirdparty_t1

    dsri = _safe_div(
        _safe_div(rec_t, data.sales_t, "DSRI rec_t/sales_t"),
        _safe_div(rec_t1, data.sales_t1, "DSRI rec_t1/sales_t1"),
        "DSRI",
    )

    # ------------------------------------------------------------------
    # GMI — Gross Margin Index
    # GMI > 1 means prior-year margin was better → deteriorating profitability.
    # Note: when BOTH margins are negative (common in loss-making companies),
    # GMI < 1 mechanically but economically margin is WORSENING — analysts
    # should inspect the absolute margin trend separately.
    # ------------------------------------------------------------------
    gross_margin_t = _safe_div(data.sales_t - data.cogs_t, data.sales_t, "GMI gm_t")
    gross_margin_t1 = _safe_div(data.sales_t1 - data.cogs_t1, data.sales_t1, "GMI gm_t1")
    gmi = _safe_div(gross_margin_t1, gross_margin_t, "GMI")

    # ------------------------------------------------------------------
    # AQI — Asset Quality Index
    # Measures growth in non-productive (intangible/deferred) assets.
    # AQI > 1 means more costs are being deferred/capitalised.
    # ------------------------------------------------------------------
    asset_quality_t = 1.0 - _safe_div(
        data.current_assets_t + data.ppe_t,
        data.total_assets_t,
        "AQI aq_t",
    )
    asset_quality_t1 = 1.0 - _safe_div(
        data.current_assets_t1 + data.ppe_t1,
        data.total_assets_t1,
        "AQI aq_t1",
    )
    aqi = _safe_div(asset_quality_t, asset_quality_t1, "AQI")

    # ------------------------------------------------------------------
    # SGI — Sales Growth Index
    # High growth companies face pressure to sustain earnings → higher risk.
    # ------------------------------------------------------------------
    sgi = _safe_div(data.sales_t, data.sales_t1, "SGI")

    # ------------------------------------------------------------------
    # DEPI — Depreciation Index
    # DEPI > 1 means depreciation rate is slowing → possible life-extension
    # manipulation to reduce depreciation expense and inflate earnings.
    # Formula: [dep_t1 / (dep_t1 + ppe_t1)] / [dep_t / (dep_t + ppe_t)]
    # Uses NET ppe consistent with Beneish original.
    # ------------------------------------------------------------------
    depi = _safe_div(
        _safe_div(
            data.depreciation_t1,
            data.depreciation_t1 + data.ppe_t1,
            "DEPI rate_t1",
        ),
        _safe_div(
            data.depreciation_t,
            data.depreciation_t + data.ppe_t,
            "DEPI rate_t",
        ),
        "DEPI",
    )

    # ------------------------------------------------------------------
    # SGAI — SG&A Expense Index
    # Uses FULL SG&A = G&A expense + selling expense, per Beneish (1999).
    # SGAI > 1 means SG&A growing disproportionately to sales.
    # ------------------------------------------------------------------
    total_sga_t = data.sga_expense_t + data.selling_expense_t
    total_sga_t1 = data.sga_expense_t1 + data.selling_expense_t1
    sgai = _safe_div(
        _safe_div(total_sga_t, data.sales_t, "SGAI sga_t/sales_t"),
        _safe_div(total_sga_t1, data.sales_t1, "SGAI sga_t1/sales_t1"),
        "SGAI",
    )

    # ------------------------------------------------------------------
    # LVGI — Leverage Index
    # Measures change in financial leverage. Rising LVGI suggests growing
    # debt pressure which may incentivise earnings manipulation.
    # Uses interest-bearing debt (current liabilities + LT debt) / TA,
    # consistent with Beneish (1999) definition.
    # ------------------------------------------------------------------
    lvgi = _safe_div(
        _safe_div(
            data.current_liabilities_t + data.long_term_debt_t,
            data.total_assets_t,
            "LVGI lev_t",
        ),
        _safe_div(
            data.current_liabilities_t1 + data.long_term_debt_t1,
            data.total_assets_t1,
            "LVGI lev_t1",
        ),
        "LVGI",
    )

    # ------------------------------------------------------------------
    # TATA — Total Accruals to Total Assets (income statement method)
    # TATA = (Operating Income - Cash Flow from Operations) / Total Assets
    # High positive TATA → large accruals → low earnings quality → red flag.
    # This method is more reliable than the balance sheet working-capital method
    # because it is unaffected by non-operating items and acquisitions.
    # ------------------------------------------------------------------
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
    """
    Beneish (1999) 8-variable M-Score formula:
    M = -4.84 + 0.920*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI
            + 0.115*DEPI - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    """
    return (
        -4.84
        + 0.920 * ratios["DSRI"]
        + 0.528 * ratios["GMI"]
        + 0.404 * ratios["AQI"]
        + 0.892 * ratios["SGI"]
        + 0.115 * ratios["DEPI"]
        - 0.172 * ratios["SGAI"]
        + 4.679 * ratios["TATA"]
        - 0.327 * ratios["LVGI"]
    )


def classify_risk(m_score: float) -> str:
    """
    Two commonly used thresholds from Beneish literature:
      > -1.78 : High risk (Beneish 1999 original threshold)
      > -2.22 : Medium risk / gray zone (more conservative threshold)
      <= -2.22 : Low risk
    """
    if m_score > -1.78:
        return "High Risk (Likely Earnings Manipulator)"
    if m_score > -2.22:
        return "Medium Risk (Gray Zone — Warrants Further Investigation)"
    return "Low Risk"


# ---------------------------------------------------------------------------
# OCR extraction helpers
# ---------------------------------------------------------------------------

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
        "selling_expense_t": None,
        "selling_expense_t1": None,
        "current_liabilities_t": None,
        "current_liabilities_t1": None,
        "long_term_debt_t": None,
        "long_term_debt_t1": None,
        "income_from_operations_t": None,
        "cash_flow_from_operations_t": None,
    }


def _parse_raw_value(raw_value: str) -> float | None:
    raw_value = raw_value.replace(" ", "")
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
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            page_count = len(doc)
            pages_to_render = min(page_count, max_pages)
            matrix = fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE)
            images: list[str] = []
            for page_index in range(pages_to_render):
                page = doc.load_page(page_index)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                images.append(base64.b64encode(pixmap.tobytes("png")).decode("ascii"))
            return images, page_count
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render PDF pages for AI OCR: {exc}") from exc


# OCR prompt generation has been moved to prompts_ocr.py


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    candidates: list[str] = []
    stripped = raw_text.strip()
    if stripped:
        candidates.append(stripped)
    fenced_match = re.search(r"```(?:json)?\s*(\{{.*?\}})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
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


def _call_openrouter_vision(prompt: str, images_b64: list[str]) -> tuple[str, dict[str, Any]]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is missing.")
    if not OPENROUTER_VISION_MODEL:
        raise HTTPException(status_code=500, detail="OPENROUTER_VISION_MODEL is missing.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    payload = {
        "model": OPENROUTER_VISION_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise financial statement OCR extraction assistant. Return only valid JSON, no markdown, no explanation."},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        error_detail = response.text if hasattr(response, "text") else str(e)
        raise HTTPException(status_code=400, detail=f"OpenRouter API Error: {error_detail}")
    data = response.json()
    usage = data.get("usage", {})
    cost_usd = _extract_openrouter_cost_usd(data)
    if isinstance(usage, dict) and cost_usd is not None:
        usage = {**usage, "estimated_cost_usd": cost_usd}
    return data["choices"][0]["message"]["content"].strip(), usage


def _call_ollama_vision(prompt: str, images_b64: list[str]) -> tuple[str, dict[str, Any]]:
    if not OLLAMA_MODEL:
        raise HTTPException(status_code=500, detail="OLLAMA_MODEL is missing.")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": images_b64}],
        "stream": False,
        "keep_alive": -1,
        "options": {"temperature": 0.0, "num_ctx": 6144},
    }
    response = requests.post(f"{base_url}/api/chat", json=payload, timeout=600)
    response.raise_for_status()
    data = response.json()
    usage = {
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "completion_tokens": data.get("eval_count", 0),
        "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
    }
    return data.get("message", {}).get("content", "").strip(), usage


def extract_financial_variables(pdf_bytes: bytes) -> tuple[dict[str, float | None], str, str, str, str, dict[str, Any]]:
    images_b64, page_count = _render_pdf_pages_for_ocr(pdf_bytes)
    processed_pages = len(images_b64)
    if not images_b64:
        raise HTTPException(status_code=400, detail="Uploaded PDF does not contain any renderable pages.")
    fields_json = json.dumps(_default_mock_variables(), ensure_ascii=True, indent=2)
    prompt = build_ocr_prompt(page_count=page_count, processed_pages=processed_pages, fields_json=fields_json)
    # Force OpenRouter for OCR to avoid 10-minute local CPU timeouts
    provider = "openrouter"
    if provider == "openrouter":
        raw_response, usage = _call_openrouter_vision(prompt, images_b64)
    else:
        raise HTTPException(status_code=500, detail="OpenRouter must be configured for OCR.")
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
        extracted_count = sum(1 for v in parsed_variables.values() if v is not None)
    note = (
        f"AI OCR extraction via {provider} completed on {processed_pages}/{page_count} page(s). "
        f"Extracted {extracted_count}/{len(parsed_variables)} variables."
    )
    model_name = OPENROUTER_MODEL if provider == "openrouter" else OLLAMA_MODEL
    return parsed_variables, note, provider, model_name or "", raw_response, usage


# ---------------------------------------------------------------------------
# RAG + LLM narrative
# ---------------------------------------------------------------------------

def query_chromadb_context(
    query_text: str,
    db_path: str | None = None,
    collection_name: str | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    db_path = db_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")
    collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "fraud_knowledge")
    try:
        import chromadb  # type: ignore
        import chromadb.utils.embedding_functions as embedding_functions

        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ollama_ef = embedding_functions.OllamaEmbeddingFunction(
            url=f"{ollama_url}/api/embeddings",
            model_name="nomic-embed-text",
        )

        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_or_create_collection(
            name=collection_name, 
            embedding_function=ollama_ef
        )
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        chunks = [
            {"content": doc, "metadata": metadatas[i] if i < len(metadatas) else {}, "distance": distances[i] if i < len(distances) else None}
            for i, doc in enumerate(docs)
        ]
        return {"source": "chromadb", "collection": collection_name, "chunks": chunks}
    except Exception as exc:
        return {
            "source": "fallback",
            "collection": collection_name,
            "chunks": [{"content": "PSAK 115 generally addresses revenue recognition rules. OJK sanction letters may indicate governance, disclosure, or reporting concerns.", "metadata": {"fallback_reason": str(exc)}, "distance": None}],
        }


# Prompt generation has been moved to prompts.py


async def stream_llm_async(prompt: str, provider_override: str | None = None):
    """Async generator that yields (kind, text) tuples.
    kind = 'thinking' for model reasoning tokens, 'chunk' for final response tokens.
    """
    provider = (provider_override or AI_PROVIDER or "").lower()
    if provider == "openrouter":
        async for kind, text in _stream_openrouter_async(prompt):
            yield kind, text
    elif provider in ("ollama", "local"):
        async for kind, text in _stream_ollama_async(prompt):
            yield kind, text
    else:
        yield "chunk", f"Layanan LLM tidak tersedia. AI_PROVIDER '{provider}' tidak didukung."

async def _stream_ollama_async(prompt: str):
    """Yields (kind, text) tuples from Ollama streaming /api/generate."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "think": False,
        "options": {"temperature": 0.3, "top_p": 0.95, "top_k": 64, "num_ctx": 6144},
    }
    in_think_tag = False  # fallback parser state for inline <think> tags


    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("POST", f"{base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)

                    # --- Structured thinking field (Ollama >= 0.7 with think=True) ---
                    thinking = data.get("thinking", "")
                    response_text = data.get("response", "")

                    if data.get("done"):
                        usage = {
                            "prompt_tokens": data.get("prompt_eval_count", 0),
                            "completion_tokens": data.get("eval_count", 0),
                            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                            "estimated_cost_usd": None
                        }
                        yield "usage", usage

                    if thinking:
                        yield "thinking", thinking
                        continue  # structured path: skip inline parsing

                    if not response_text:
                        continue

                    # --- Fallback: parse <think>...</think> inline in response tokens ---
                    # Accumulate token into buffer and scan for open/close tags
                    buf = response_text
                    while buf:
                        if in_think_tag:
                            end = buf.find("</think>")
                            if end == -1:
                                yield "thinking", buf
                                buf = ""
                            else:
                                if end > 0:
                                    yield "thinking", buf[:end]
                                in_think_tag = False
                                buf = buf[end + len("</think>"):]
                        else:
                            start = buf.find("<think>")
                            if start == -1:
                                yield "chunk", buf
                                buf = ""
                            else:
                                if start > 0:
                                    yield "chunk", buf[:start]
                                in_think_tag = True
                                buf = buf[start + len("<think>"):]
    except Exception as exc:
        yield "chunk", f" Layanan LLM tidak tersedia. Detail error: {exc}"


async def _stream_openrouter_async(prompt: str):
    """Yields (kind, text) tuples from OpenRouter streaming.
    OpenRouter also supports reasoning tokens via 'delta.reasoning' for models
    like DeepSeek-R1; we forward those as 'thinking' kind.
    """
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        yield "chunk", "OpenRouter API key belum diatur."
        return
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "Selalu jawab dalam Bahasa Indonesia profesional."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    text = line.strip()
                    if text.startswith("data: "):
                        text = text[6:]
                    if text == "[DONE]":
                        break
                    try:
                        data = json.loads(text)
                        
                        if "usage" in data and data["usage"]:
                            cost_usd = _extract_openrouter_cost_usd(data)
                            usage_dict = {
                                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                                "completion_tokens": data["usage"].get("completion_tokens", 0),
                                "total_tokens": data["usage"].get("total_tokens", 0),
                                "estimated_cost_usd": cost_usd
                            }
                            yield "usage", usage_dict

                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                            content = delta.get("content", "")
                            if reasoning:
                                yield "thinking", reasoning
                            if content:
                                yield "chunk", content
                    except Exception:
                        pass
    except Exception as exc:
        yield "chunk", f" Layanan LLM tidak tersedia. Detail error: {exc}"

def call_llm(prompt: str, provider_override: str | None = None) -> tuple[str, str, str, str, dict[str, Any] | None]:
    provider = (provider_override or AI_PROVIDER or "").lower()
    if provider == "openrouter":
        narrative, model_provider, model_name, raw_response, usage = _call_openrouter(prompt)
    elif provider in ("ollama", "local"):
        narrative, model_provider, model_name, raw_response, usage = _call_ollama(prompt)
    else:
        fallback = f"Layanan LLM tidak tersedia. AI_PROVIDER '{provider}' tidak didukung."
        return fallback, provider, "", json.dumps({"error": "Unknown provider"}, ensure_ascii=True), None
    if _needs_indonesian_rewrite(narrative):
        rewritten_text, rewrite_raw = _rewrite_to_bahasa_indonesia(narrative, provider)
        if rewritten_text:
            narrative = rewritten_text
            raw_response = json.dumps({"initial_raw_response": raw_response, "rewrite_raw_response": rewrite_raw}, ensure_ascii=True)
    return narrative, model_provider, model_name, raw_response, usage


def _needs_indonesian_rewrite(text: str) -> bool:
    normalized = f" {text.lower()} "
    english_markers = [" the ", " and ", " with ", " company ", " revenue ", " risk ", " recommend ", " therefore "]
    indonesian_markers = [" dan ", " yang ", " dengan ", " terhadap ", " perusahaan ", " pendapatan ", " risiko ", " rekomendasi "]
    return sum(1 for m in english_markers if m in normalized) > sum(1 for m in indonesian_markers if m in normalized)


def _rewrite_to_bahasa_indonesia(text: str, provider: str) -> tuple[str | None, str]:
    rewrite_prompt = (
        "Tulis ulang teks berikut seluruhnya dalam Bahasa Indonesia profesional. "
        "Jangan ubah fakta, angka, atau kesimpulan. Hanya ubah bahasanya.\n\n"
        f"Teks:\n{text}"
    )
    try:
        if provider == "openrouter":
            rewritten, _, _, raw, _ = _call_openrouter(rewrite_prompt)
            return rewritten, raw
        rewritten, _, _, raw, _ = _call_ollama(rewrite_prompt)
        return rewritten, raw
    except Exception as exc:
        return None, json.dumps({"rewrite_error": str(exc)}, ensure_ascii=True)


def _call_ollama(prompt: str) -> tuple[str, str, str, str, dict[str, Any] | None]:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "top_p": 0.95, "top_k": 64, "num_ctx": 6144},
    }
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
        usage = {
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            "estimated_cost_usd": None,
        }
        return data.get("response", "No response returned by Ollama.").strip(), "ollama", OLLAMA_MODEL or "", json.dumps(data, ensure_ascii=True), usage
    except Exception as exc:
        fallback = f"Layanan LLM tidak tersedia. Detail error: {exc}"
        return fallback, "ollama", OLLAMA_MODEL or "", json.dumps({"error": str(exc)}, ensure_ascii=True), None


def _call_openrouter(prompt: str) -> tuple[str, str, str, str, dict[str, Any] | None]:
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        message = "OpenRouter API key belum diatur. Silakan set OPENROUTER_API_KEY di environment."
        return message, "openrouter", OPENROUTER_MODEL or "", json.dumps({"error": message}, ensure_ascii=True), None
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
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
        usage = data.get("usage", {})
        cost_usd = _extract_openrouter_cost_usd(data)
        if isinstance(usage, dict) and cost_usd is not None:
            usage = {**usage, "estimated_cost_usd": cost_usd}
        return data["choices"][0]["message"]["content"].strip(), "openrouter", OPENROUTER_MODEL or "", json.dumps(data, ensure_ascii=True), usage if isinstance(usage, dict) else None
    except Exception as exc:
        fallback = f"Layanan LLM tidak tersedia. Detail error: {exc}"
        return fallback, "openrouter", OPENROUTER_MODEL or "", json.dumps({"error": str(exc)}, ensure_ascii=True), None


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

def _check_ai_connection(provider: str) -> str:
    try:
        if provider == "openrouter":
            if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
                return "error (missing API key)"
            headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
            resp = requests.get("https://openrouter.ai/api/v1/auth/key", headers=headers, timeout=3)
            return "ok" if resp.status_code == 200 else f"error (HTTP {resp.status_code})"
        elif provider in ("ollama", "local"):
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            resp = requests.get(f"{base_url}/api/tags", timeout=3)
            return "ok" if resp.status_code == 200 else f"error (HTTP {resp.status_code})"
        else:
            return f"error (unknown provider: {provider})"
    except Exception as exc:
        return f"error ({type(exc).__name__}: {exc})"

@app.get("/health")
def health() -> dict[str, Any]:
    started_at = time.perf_counter()
    chroma_status = "error"
    doc_count = 0
    try:
        import chromadb
        db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "fraud_knowledge")
        client = chromadb.PersistentClient(path=db_path)
        collection = client.get_collection(name=collection_name)
        doc_count = collection.count()
        chroma_status = "ok"
    except Exception:
        pass

    provider = (AI_PROVIDER or "").strip().lower()
    model_name = OPENROUTER_MODEL if provider == "openrouter" else OLLAMA_MODEL
    ai_status = _check_ai_connection(provider)

    return {
        "status": "ok",
        "ai_provider": provider,
        "ai_model": model_name,
        "ai_connection_status": ai_status,
        "chromadb_status": chroma_status,
        "knowledge_documents_count": doc_count,
        "responded_at": _iso_utc_now(),
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 3)
    }


@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    started_at = time.perf_counter()
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    extracted_variables, extraction_note, ocr_provider, ocr_model, ocr_raw_response, ocr_usage = extract_financial_variables(pdf_bytes)
    ocr_estimated_cost_usd = _to_float_or_none(ocr_usage.get("estimated_cost_usd")) if isinstance(ocr_usage, dict) else None
    preview = (
        f"AI OCR processed {file.filename or 'uploaded.pdf'} and extracted "
        f"{sum(1 for v in extracted_variables.values() if v is not None)} financial fields."
    )
    return UploadResponse(
        filename=file.filename or "uploaded.pdf",
        extraction_note=extraction_note,
        extracted_text_preview=preview,
        extracted_variables=extracted_variables,
        ocr_provider=ocr_provider,
        ocr_model=ocr_model,
        ocr_raw_response=ocr_raw_response,
        ocr_usage=ocr_usage,
        ocr_estimated_cost_usd=ocr_estimated_cost_usd,
        responded_at=_iso_utc_now(),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
    )


@app.post("/api/analyze")
def analyze_validated_data(
    payload: FinancialVariables = Body(...),
):
    started_at = time.perf_counter()

    # Step A — Compute Beneish ratios + M-Score
    ratios = calculate_beneish_ratios(payload)
    m_score = calculate_m_score(ratios)
    risk_status = classify_risk(m_score)

    # Step B — Retrieve RAG context
    rag_query = f"Beneish M-Score {m_score:.4f}, risk {risk_status}, PSAK 115 OJK sanction"
    rag_result = query_chromadb_context(rag_query)
    rag_chunks = rag_result.get("chunks", [])

    # Step C — LLM narrative
    prompt = build_llm_prompt(payload.company_name, payload.fiscal_year, ratios, m_score, risk_status, rag_chunks)
    
    def event_generator():
        # Yield metadata first
        metadata = {
            "ratios": {k: round(v, 6) for k, v in ratios.items()},
            "m_score": round(m_score, 6),
            "risk_status": risk_status,
            "llm_provider": "ollama",
            "llm_model": OLLAMA_MODEL or "",
        }
        yield f"data: {json.dumps({'type': 'metadata', 'data': metadata})}\n\n"
        
        # Yield text chunks
        try:
            for chunk in stream_llm(prompt, provider_override="ollama"):
                if chunk:
                    yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
            
        duration = round((time.perf_counter() - started_at) * 1000, 3)
        yield f"data: {json.dumps({'type': 'done', 'duration_ms': duration})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.websocket("/api/ws/analyze")
async def analyze_ws(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        payload = FinancialVariables(**data)

        started_at = time.perf_counter()

        await websocket.send_json({"type": "status", "message": "Computing Beneish M-Score..."})

        # Step A — Compute Beneish ratios + M-Score
        ratios = calculate_beneish_ratios(payload)
        m_score = calculate_m_score(ratios)
        risk_status = classify_risk(m_score)

        await websocket.send_json({"type": "status", "message": "Retrieving RAG knowledge base..."})

        # Step B — Retrieve RAG context
        # Run blocking chromadb query in a threadpool so it doesn't block the async event loop
        rag_query = f"Beneish M-Score {m_score:.4f}, risk {risk_status}, PSAK 115 OJK sanction"
        rag_result = await asyncio.to_thread(query_chromadb_context, rag_query)
        rag_chunks = rag_result.get("chunks", [])

    # Step C — LLM narrative
        prompt = build_llm_prompt(payload.company_name, payload.fiscal_year, ratios, m_score, risk_status, rag_chunks)

        # --- Keepalive ping task ---
        # Starts BEFORE the lock so the user gets pings while waiting in the queue!
        stop_ping = asyncio.Event()
        is_generating = False

        async def _keepalive():
            cool_messages = [
                "Processing financial variables...",
                "Cross-referencing Beneish Ratios with PSAK 115...",
                "Analyzing Step 5 Revenue Recognition anomalies...",
                "Evaluating Beneish M-Score red flags...",
                "Synthesizing forensic accounting narrative...",
                "AI is thinking in progress...",
                "Just a moment, the AI is still processing...",
            ]
            idx = 0
            while not stop_ping.is_set():
                try:
                    await asyncio.wait_for(asyncio.shield(stop_ping.wait()), timeout=8.0)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"type": "ping"})
                        if is_generating and not stop_ping.is_set():
                            msg = cool_messages[idx % len(cool_messages)]
                            await websocket.send_json({"type": "status", "message": msg})
                            idx += 1
                    except Exception:
                        break

        ping_task = asyncio.create_task(_keepalive())

        llm_usage = None
        try:
            if llm_queue_lock.locked():
                await websocket.send_json({"type": "status", "message": "Waiting in queue... (Server is busy)"})
                
            async with llm_queue_lock:
                is_generating = True
                await websocket.send_json({"type": "status", "message": "Applying Core PSAK 115 Principles..."})
                
                # The /analyze endpoint is forced to 'ollama' in this specific flow.
                provider = "ollama"
                llm_model = OLLAMA_MODEL or ""

                # Yield metadata first (only right before generation)
                metadata = {
                    "type": "metadata",
                    "data": {
                        "ratios": {k: round(v, 6) for k, v in ratios.items()},
                        "m_score": round(m_score, 6),
                        "risk_status": risk_status,
                        "llm_provider": provider,
                        "llm_model": llm_model,
                    },
                }
                await websocket.send_json(metadata)

                # Yield text chunks
                async for kind, content in stream_llm_async(prompt):
                    if kind == "usage":
                        llm_usage = content
                    elif content:
                        await websocket.send_json({"type": kind, "text": content})
        except Exception as e:
            await websocket.send_json({"type": "error", "text": str(e)})
        finally:
            stop_ping.set()
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

        duration = round((time.perf_counter() - started_at) * 1000, 3)
        await websocket.send_json({"type": "done", "duration_ms": duration, "usage": llm_usage})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "text": str(e)})
        except Exception:
            pass


@app.post("/api/analyze-calk", response_model=AnalyzeCalkResponse)
async def analyze_calk(
    file: UploadFile = File(...),
    m_score: float = Form(...),
    risk_status: str = Form(...),
    ratios_json: str = Form(...),
    extracted_variables_json: str = Form(default="{}"),
) -> AnalyzeCalkResponse:
    started_at = time.perf_counter()
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")

    try:
        import fitz  # type: ignore
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text_content = ""
            # Extract text from all pages since CaLK notes can be deep in the document
            for page in doc:
                text_content += page.get_text() + "\n"
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from CaLK PDF: {exc}")

    # Truncate text to approx 400,000 characters (roughly 100k tokens) to fit in modern context windows safely
    text_content = text_content[:400000]

    # Retrieve RAG context (PSAK 115)
    rag_query = f"Beneish M-Score {m_score:.4f}, risk {risk_status}, PSAK 115 OJK sanction"
    rag_result = query_chromadb_context(rag_query)
    rag_chunks = rag_result.get("chunks", [])
    
    prompt = build_calk_prompt(
        m_score=m_score,
        risk_status=risk_status,
        ratios_json=ratios_json,
        extracted_variables_json=extracted_variables_json,
        rag_chunks=rag_chunks,
        text_content=text_content,
    )

    llm_narrative, llm_provider, llm_model, llm_raw_response, llm_usage = call_llm(prompt, provider_override="openrouter")

    return AnalyzeCalkResponse(
        deep_analysis_insight=llm_narrative,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_raw_response=llm_raw_response,
        llm_usage=llm_usage,
        responded_at=_iso_utc_now(),
        duration_ms=round((time.perf_counter() - started_at) * 1000, 3)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
