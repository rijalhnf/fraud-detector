from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

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


class UploadResponse(BaseModel):
    filename: str
    extraction_note: str
    extracted_text_preview: str
    extracted_variables: dict[str, float | None]
    ocr_provider: str | None = None
    ocr_model: str | None = None
    ocr_raw_response: str | None = None
    ocr_usage: dict[str, Any] | None = None
    ocr_estimated_cost_usd: float | None = None
    responded_at: str
    duration_ms: float


class FinancialVariables(BaseModel):
    """
    User-validated financial variables required for Beneish M-Score calculation.

    Naming convention:
      _t  = current year (year under analysis)
      _t1 = prior year (one year before)

    Key methodology notes:
      - receivables_t/t1  : TRADE receivables ONLY (piutang usaha: related party + third party).
                            Do NOT include piutang lain-lain (other receivables).
      - sga_expense_t/t1  : General & Administrative expenses only (beban umum dan administrasi).
      - selling_expense_t/t1 : Selling expenses separately (beban penjualan / beban tiket, penjualan
                            dan promosi). These are ADDED to sga_expense in SGAI calculation per
                            original Beneish (1999) which uses full SG&A.
      - depreciation_t/t1 : Actual depreciation charge for the year. For companies using the
                            revaluation model on PPE (e.g. airlines), this MUST come from the
                            Notes to Financial Statements (fixed assets note), NOT from the
                            delta of accumulated depreciation on the balance sheet.
      - ppe_t/t1          : Net PPE (after accumulated depreciation), as shown on balance sheet.
      - current_liabilities_t/t1 : Total current liabilities (all short-term obligations).
      - long_term_debt_t/t1 : Interest-bearing long-term debt only (bank loans + bonds +
                            finance lease liabilities). Exclude non-debt liabilities like
                            deferred revenue, employee benefits, provisions.
      - income_from_operations_t  : Operating profit/loss (laba/rugi usaha), before finance
                            income/cost and tax.
      - cash_flow_from_operations_t : Net cash from operating activities (direct or indirect method).
      - cogs_t/t1         : For airlines/service companies without explicit COGS, use total
                            operating expenses (beban usaha) as a proxy.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "receivables_t": 414100677.0,
                "receivables_t1": 229250088.0,
                "sales_t": 4373177070.0,
                "sales_t1": 4177325781.0,
                "cogs_t": 4579259674.0,
                "cogs_t1": 4237773332.0,
                "current_assets_t": 1356974740.0,
                "current_assets_t1": 986741627.0,
                "ppe_t": 944002399.0,
                "ppe_t1": 900657607.0,
                "total_assets_t": 4371659686.0,
                "total_assets_t1": 3763292093.0,
                "depreciation_t": 177964733.0,
                "depreciation_t1": 143312108.0,
                "sga_expense_t": 221343549.0,
                "sga_expense_t1": 265808770.0,
                "selling_expense_t": 324376515.0,
                "selling_expense_t1": 323723174.0,
                "current_liabilities_t": 2451116662.0,
                "current_liabilities_t1": 1921846147.0,
                "long_term_debt_t": 130282434.0,
                "long_term_debt_t1": 129251212.0,
                "income_from_operations_t": 100801326.0,
                "cash_flow_from_operations_t": 270751794.0,
                "company_name": "PT Garuda Indonesia (Persero) Tbk",
                "fiscal_year": "2018",
            }
        },
    )

    # --- DSRI inputs ---
    receivables_t: float = Field(
        ...,
        description=(
            "Net TRADE receivables current year. "
            "Include: piutang usaha pihak berelasi + piutang usaha pihak ketiga. "
            "EXCLUDE: piutang lain-lain (other receivables), piutang pajak, uang muka."
        ),
    )
    receivables_t1: float = Field(
        ...,
        description=(
            "Net TRADE receivables prior year. Same definition as receivables_t."
        ),
    )
    receivables_related_t: float = Field(
    default=0.0,
    description="Trade receivables — related parties only (piutang usaha pihak berelasi).",
    )
    receivables_related_t1: float = Field(default=0.0)
    receivables_thirdparty_t: float = Field(
        default=0.0,
        description="Trade receivables — third parties only (piutang usaha pihak ketiga, net of allowance).",
    )
    receivables_thirdparty_t1: float = Field(default=0.0)

    # --- GMI & SGI inputs ---
    sales_t: float = Field(..., description="Net revenue/sales current year (total pendapatan usaha).")
    sales_t1: float = Field(..., description="Net revenue/sales prior year.")

    cogs_t: float = Field(
        ...,
        description=(
            "Cost of goods sold current year. "
            "For airlines/service companies: use total operating expenses (total beban usaha) as proxy."
        ),
    )
    cogs_t1: float = Field(
        ...,
        description="Cost of goods sold prior year. Same proxy rule as cogs_t.",
    )

    # --- AQI inputs ---
    current_assets_t: float = Field(..., description="Total current assets current year (total aset lancar).")
    current_assets_t1: float = Field(..., description="Total current assets prior year.")

    ppe_t: float = Field(
        ...,
        description=(
            "Net PPE current year (aset tetap setelah dikurangi akumulasi penyusutan). "
            "Use the NET book value as reported on the balance sheet."
        ),
    )
    ppe_t1: float = Field(..., description="Net PPE prior year.")

    total_assets_t: float = Field(..., description="Total assets current year (total aset).")
    total_assets_t1: float = Field(..., description="Total assets prior year.")

    # --- DEPI inputs ---
    depreciation_t: float = Field(
        ...,
        description=(
            "Actual depreciation expense charged for current year. "
            "IMPORTANT: For companies using the PPE revaluation model (common in airlines), "
            "this value MUST come from the Notes to Financial Statements (fixed assets note / "
            "catatan aset tetap), NOT from delta of accumulated depreciation on the balance sheet. "
            "Look for text like 'Beban penyusutan yang dibebankan dalam operasi sebesar USD X' "
            "or 'Depreciation expense charged to operations amounted to USD X'."
        ),
    )
    depreciation_t1: float = Field(
        ...,
        description="Actual depreciation expense prior year. Same source guidance as depreciation_t.",
    )

    # --- SGAI inputs (split into G&A + Selling for accuracy) ---
    sga_expense_t: float = Field(
        ...,
        description=(
            "General & Administrative expense current year ONLY "
            "(beban umum dan administrasi / beban administrasi dan umum). "
            "Do NOT include selling expenses here — they go in selling_expense_t."
        ),
    )
    sga_expense_t1: float = Field(
        ...,
        description="General & Administrative expense prior year.",
    )
    selling_expense_t: float = Field(
        default=0.0,
        description=(
            "Selling expenses current year (beban penjualan, beban tiket penjualan dan promosi, "
            "beban pemasaran). Per Beneish (1999), SGAI uses full SG&A, so this is added to "
            "sga_expense_t in the calculation. Set to 0.0 if not separately disclosed."
        ),
    )
    selling_expense_t1: float = Field(
        default=0.0,
        description="Selling expenses prior year. Same definition as selling_expense_t.",
    )

    # --- LVGI inputs ---
    current_liabilities_t: float = Field(
        ...,
        description="Total current liabilities current year (total liabilitas jangka pendek).",
    )
    current_liabilities_t1: float = Field(
        ...,
        description="Total current liabilities prior year.",
    )
    long_term_debt_t: float = Field(
        ...,
        description=(
            "Interest-bearing long-term debt current year (net of current maturities). "
            "Include: pinjaman jangka panjang + liabilitas sewa pembiayaan + utang obligasi. "
            "EXCLUDE: pendapatan diterima dimuka, liabilitas imbalan kerja, provisi, "
            "liabilitas pajak tangguhan (these are non-debt obligations)."
        ),
    )
    long_term_debt_t1: float = Field(
        ...,
        description="Interest-bearing long-term debt prior year. Same definition as long_term_debt_t.",
    )

    # --- TATA inputs ---
    income_from_operations_t: float = Field(
        ...,
        description=(
            "Operating income current year (laba/rugi usaha). "
            "This is BEFORE finance income, finance cost, and tax expense. "
            "Look for 'Laba (Rugi) Usaha' or 'Profit (Loss) from Operations'."
        ),
    )
    cash_flow_from_operations_t: float = Field(
        ...,
        description=(
            "Net cash from operating activities current year "
            "(kas bersih diperoleh dari/digunakan untuk aktivitas operasi). "
            "From the Statement of Cash Flows (Exhibit D / Laporan Arus Kas)."
        ),
    )

    company_name: str | None = Field(default=None)
    fiscal_year: str | None = Field(default=None)


class AnalyzeResponse(BaseModel):
    ratios: dict[str, float]
    m_score: float
    risk_status: str
    llm_narrative_insight: str
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_raw_response: str | None = None
    llm_usage: dict[str, Any] | None = None
    llm_estimated_cost_usd: float | None = None
    responded_at: str
    duration_ms: float


class AnalyzeCalkResponse(BaseModel):
    deep_analysis_insight: str
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_raw_response: str | None = None
    llm_usage: dict[str, Any] | None = None
    responded_at: str
    duration_ms: float


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


def _build_ocr_prompt(page_count: int, processed_pages: int) -> str:
    """
    Comprehensive OCR extraction prompt for financial statement PDFs.
    Covers Indonesian (PSAK) and international (IFRS/GAAP) terminology,
    with field-by-field extraction rules and common pitfall warnings.
    """
    fields_json = json.dumps(_default_mock_variables(), ensure_ascii=True, indent=2)

    return f"""
You are a forensic accounting OCR engine specialising in extracting Beneish M-Score inputs from financial statement PDFs.
The document has {page_count} page(s). You are processing the first {processed_pages} page(s).

Return ONLY a single valid JSON object matching the schema below. No markdown fences, no explanation text, no preamble.
All numeric values must be plain JSON numbers (not strings). Use null for any field you cannot find.
Normalise all number formats: remove thousand-separators (dots or commas used as thousands), keep the decimal separator as a period.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATEMENT STRUCTURE TO LOOK FOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Financial statements typically contain these exhibits/schedules:
  • Exhibit A / Laporan Posisi Keuangan  →  Balance Sheet (assets, liabilities, equity)
  • Exhibit B / Laporan Laba Rugi         →  Income Statement (revenue, expenses, profit)
  • Exhibit D / Laporan Arus Kas          →  Cash Flow Statement
  • Exhibit E / Catatan                   →  Notes to Financial Statements

Column structure: the FIRST numeric column is the CURRENT year (_t), the SECOND column is the PRIOR year (_t1).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD-BY-FIELD EXTRACTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[receivables_t / receivables_t1] — TRADE RECEIVABLES ONLY
  The balance sheet often shows trade receivables split into TWO sub-lines under one header:
    Line 1 → Piutang usaha pihak berelasi (related parties)
    Line 2 → Piutang usaha pihak ketiga (third parties, net of allowance)

  ✘ EXCLUDE (do NOT add these to the sum):
      - Piutang lain-lain / Other receivables — these are NON-TRADE
      - Piutang pajak / Tax receivables
      - Uang muka / Advance payments
      - Piutang karyawan / Employee loans

  You MUST extract ALL FOUR values separately:
    "receivables_related_t"    → related party, current year
    "receivables_related_t1"   → related party, prior year
    "receivables_thirdparty_t" → third party net, current year
    "receivables_thirdparty_t1"→ third party net, prior year

  Also set "receivables_t" = related + third party (their sum, current year)
  Also set "receivables_t1" = related + third party (their sum, prior year)

  NEVER set receivables_t to only one of the two sub-lines.

[sales_t / sales_t1] — REVENUE
  ✔ Total pendapatan usaha / Total operating revenues / Net sales
  → Use the TOTAL revenue line, not sub-components.

[cogs_t / cogs_t1] — COST OF GOODS SOLD
  ✔ Beban pokok pendapatan / Beban pokok penjualan / Cost of revenue / COGS
  → For AIRLINES and SERVICE companies that do NOT have an explicit COGS line:
     Use TOTAL BEBAN USAHA (total operating expenses) as a proxy.
     This is the sum of all operating expense line items before "other income/expense".
  → Do NOT use net loss or total comprehensive loss.

[current_assets_t / current_assets_t1] — TOTAL CURRENT ASSETS
  ✔ Total aset lancar / Total current assets
  → The sub-total line on the balance sheet, not individual items.

[ppe_t / ppe_t1] — NET PROPERTY, PLANT & EQUIPMENT
  ✔ Aset tetap — setelah dikurangi akumulasi penyusutan / Fixed assets — net of accumulated depreciation
  → Use the NET value (after depreciation), NOT the gross value.
  → For airlines: this is typically the largest non-current asset item.

[total_assets_t / total_assets_t1] — TOTAL ASSETS
  ✔ Total aset / Total assets
  → The grand total at the bottom of the assets section.

[depreciation_t / depreciation_t1] — DEPRECIATION EXPENSE (CRITICAL)
  ⚠ This is the MOST COMMON extraction error. Follow these rules strictly:

  PREFERRED SOURCE — Notes to Financial Statements (Catatan/Exhibit E):
    Look for the fixed assets note. It contains a sentence like:
      "Beban penyusutan yang dibebankan dalam operasi sebesar USD X (2017: USD Y)"
      "Depreciation expense charged to operations amounted to USD X (prior year: USD Y)"
    This is the EXACT annual depreciation charge. Use this value.

  ALTERNATIVE SOURCE — Cash Flow Statement (if using indirect method):
    If the cash flow statement uses the INDIRECT method, depreciation appears as a
    non-cash add-back line:
      "Penyusutan dan amortisasi / Depreciation and amortisation"
    This is also acceptable.

  ✘ DO NOT USE — Balance Sheet Delta:
    For companies using the PPE REVALUATION MODEL (common in airlines, property companies),
    the change in accumulated depreciation on the balance sheet does NOT equal the
    depreciation expense, because revaluation resets the accumulated depreciation figure.
    Never compute depreciation as: accum_dep_t minus accum_dep_t1.

  ✘ DO NOT USE — Direct method cash flow statements:
    If the cash flow uses the DIRECT method (shows cash receipts/payments directly
    without a reconciliation section), depreciation will NOT appear there.
    In that case, you MUST find it in the Notes.

[sga_expense_t / sga_expense_t1] — G&A EXPENSES ONLY
  ✔ Beban umum dan administrasi / Beban administrasi dan umum
  ✔ General and administrative expenses / Administrative expenses
  → Extract ONLY the G&A line. Selling expenses go in selling_expense_t separately.

[selling_expense_t / selling_expense_t1] — SELLING EXPENSES ONLY
  ✔ Beban penjualan / Beban tiket, penjualan dan promosi / Selling expenses
  ✔ Beban pemasaran / Marketing expenses / Distribution expenses
  ✔ For airlines: "Beban tiket, penjualan dan promosi" is the selling expense line.
  → If no separate selling expense line exists, set to 0.
  → Do NOT double-count items already in sga_expense.

[current_liabilities_t / current_liabilities_t1] — TOTAL CURRENT LIABILITIES
  ✔ Total liabilitas jangka pendek / Total current liabilities
  → The sub-total line. Include ALL current liabilities (trade payables, short-term loans,
    current portion of long-term debt, accruals, deferred revenue, tax payable, etc.)

[long_term_debt_t / long_term_debt_t1] — INTEREST-BEARING LONG-TERM DEBT ONLY
  ✔ Include (net of current maturities, i.e. the non-current portion only):
      - Pinjaman jangka panjang / Long-term loans / Long-term bank borrowings
      - Liabilitas sewa pembiayaan / Finance lease liabilities
      - Utang obligasi / Bonds payable / Notes payable
  ✘ EXCLUDE:
      - Pendapatan diterima dimuka / Deferred revenue (not financial debt)
      - Liabilitas imbalan kerja / Employee benefit obligations
      - Provisi / Provisions
      - Liabilitas pajak tangguhan / Deferred tax liabilities
      - Liabilitas estimasi biaya pengembalian / Aircraft return cost provisions
  → Sum ONLY the three debt instrument lines listed under non-current liabilities.

[income_from_operations_t] — OPERATING PROFIT / LOSS
  ✔ Laba (rugi) usaha / Profit (loss) from operations / Operating income (loss)
  → This is BEFORE: finance income, finance cost (beban keuangan), tax expense.
  → Look for the subtotal line after all operating expense items but before "other income/expense".
  → Can be negative (operating loss) — that is a valid value.

[cash_flow_from_operations_t] — NET CASH FROM OPERATING ACTIVITIES
  ✔ Kas bersih diperoleh dari (digunakan untuk) aktivitas operasi
  ✔ Net cash provided by (used in) operating activities
  → From the Cash Flow Statement (Exhibit D / Laporan Arus Kas).
  → Can be negative — that is a valid value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INDUSTRY-SPECIFIC GUIDANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Airlines (e.g. Garuda Indonesia, AirAsia, Lion Air):
  - No explicit COGS → use total beban usaha (total operating expenses)
  - Large maintenance reserve funds are NOT PPE → do not include in ppe_t
  - Finance lease liabilities on aircraft ARE long-term debt → include in long_term_debt_t
  - Uang muka pembelian pesawat (advance payments for aircraft) is NOT PPE (it is a non-current asset)
  - Depreciation will almost always be in Notes (revaluation model is common)

Banks & Insurance companies:
  - Beneish M-Score is NOT applicable to banks or insurance companies
  - Do not attempt extraction for these industries

Manufacturing companies:
  - COGS is typically explicitly stated
  - Depreciation often appears in both income statement notes and cash flow reconciliation

Retail/Trading companies:
  - COGS = Harga pokok penjualan (explicit line item)
  - Selling expenses are usually large relative to G&A

Property/Real Estate companies:
  - PPE revaluation model is common → get depreciation from Notes
  - Investment properties may be separate from PPE → do not include in ppe_t

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NUMBER FORMAT NORMALISATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Indonesian statements often use period as thousands separator and comma as decimal:
  "1.356.974.740" → 1356974740  (period = thousands separator, no decimal)
  "76.888.013"    → 76888013
  "944.002.399"   → 944002399
  "0,0003"        → 0.0003      (comma = decimal separator)

International/USD statements may use comma as thousands:
  "1,356,974,740" → 1356974740
  "177,964,733"   → 177964733

Always output as a plain JSON number with no separators.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{fields_json}
""".strip()


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
    if not OPENROUTER_MODEL:
        raise HTTPException(status_code=500, detail="OPENROUTER_MODEL is missing.")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_b64 in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a precise financial statement OCR extraction assistant. Return only valid JSON, no markdown, no explanation."},
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    response.raise_for_status()
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
        "options": {"temperature": 0.0, "num_ctx": 8192},
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
    prompt = _build_ocr_prompt(page_count=page_count, processed_pages=processed_pages)
    provider = (AI_PROVIDER or "").strip().lower()
    if provider == "openrouter":
        raw_response, usage = _call_openrouter_vision(prompt, images_b64)
    elif provider in ("ollama", "local"):
        raw_response, usage = _call_ollama_vision(prompt, images_b64)
    else:
        raise HTTPException(status_code=500, detail="AI_PROVIDER must be 'openrouter', 'ollama', or 'local'.")
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
    top_k: int = 4,
) -> dict[str, Any]:
    db_path = db_path or os.getenv("CHROMA_DB_PATH", "./chroma_db")
    collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "fraud_knowledge")
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
            "note": "Dihitung dari piutang usaha (trade receivables) saja, bukan piutang lain-lain.",
        },
        "GMI": {
            "non_manipulator_mean": 1.014,
            "red_flag_threshold": 1.193,
            "insight": "Margin memburuk dapat mendorong manajemen menyamarkan kerugian.",
            "note": "Jika kedua margin negatif (perusahaan rugi), GMI < 1 secara mekanis namun margin ekonomis sebenarnya memburuk — analisis tren absolut margin diperlukan.",
        },
        "AQI": {
            "non_manipulator_mean": 1.039,
            "red_flag_threshold": 1.254,
            "insight": "Kualitas aset menurun akibat deferral atau kapitalisasi biaya agresif.",
        },
        "SGI": {
            "non_manipulator_mean": 1.134,
            "red_flag_threshold": 1.607,
            "insight": "Pertumbuhan tinggi meningkatkan tekanan untuk mempertahankan ekspektasi pasar.",
        },
        "DEPI": {
            "non_manipulator_mean": 1.001,
            "red_flag_threshold": 1.077,
            "insight": "Perlambatan penyusutan dapat menaikkan laba secara artifisial (perpanjangan umur aset).",
            "note": "Menggunakan beban penyusutan aktual dari Catatan Laporan Keuangan, bukan delta akumulasi penyusutan.",
        },
        "SGAI": {
            "non_manipulator_mean": 1.054,
            "red_flag_threshold": 1.041,
            "insight": "Efisiensi beban SG&A melemah terhadap pendapatan.",
            "note": "Dihitung dari full SG&A = beban administrasi + beban penjualan, sesuai Beneish (1999).",
        },
        "LVGI": {
            "non_manipulator_mean": 1.037,
            "red_flag_threshold": 1.111,
            "insight": "Leverage naik meningkatkan risiko manipulasi untuk memenuhi covenant utang.",
            "note": "Menggunakan utang berbunga (liabilitas jangka pendek + utang jangka panjang berbunga), bukan total liabilitas.",
        },
        "TATA": {
            "non_manipulator_mean": 0.018,
            "red_flag_threshold": 0.031,
            "insight": "Akrual tinggi relatif terhadap arus kas mengindikasikan kualitas laba rendah.",
            "note": "Metode income statement: (Laba Usaha - Arus Kas Operasi) / Total Aset.",
        },
    }

    context_lines = [
        f"[{idx}] {chunk.get('content', '')}\nMetadata: {json.dumps(chunk.get('metadata', {}), ensure_ascii=True)}"
        for idx, chunk in enumerate(rag_chunks, start=1)
    ]

    company = data.company_name or "Unknown Company"
    year = data.fiscal_year or "Unknown Fiscal Year"

    return f"""
You are a forensic accounting assistant for an AI Fraud Early Warning System.
Analyze the company's fraud risk using Beneish M-Score results and regulatory/accounting context.

Company: {company}
Fiscal Year: {year}

Calculated Beneish Ratios (8-variable model, Beneish 1999):
{json.dumps(ratios, indent=2)}

Final M-Score: {m_score:.4f}
Risk Status: {risk_status}

Threshold reference:
  > -1.78  : High Risk (Beneish 1999 original)
  > -2.22  : Medium Risk / Gray Zone (conservative threshold)
  <= -2.22 : Low Risk

Retrieved Context (RAG from PSAK 115, before it's named PSAK 72 mandatory changed per 1 January 2024, and OJK sanction-related references):
{chr(10).join(context_lines)}

Beneish Benchmark Reference (per ratio red flags and methodology notes):
{json.dumps(benchmark_reference, ensure_ascii=True, indent=2)}

Instructions:
1) Explain what the M-Score value means for this company in plain but professional language.
2) Identify which ratios exceed their red-flag thresholds and explain the forensic implication of each.
3) Note any ratios where the mechanical result may be misleading (e.g. GMI when both margins are negative).
4) PSAK 115 / PSAK 72 Analysis — map each red-flag ratio to the relevant
   revenue recognition step:
     Step 1 (Identify contract)      → related to DSRI anomalies
     Step 2 (Identify obligations)   → related to deferred revenue changes
     Step 3 (Determine price)        → related to GMI / variable consideration
     Step 4 (Allocate price)         → related to multi-element arrangements
     Step 5 (Recognize revenue)      → related to DSRI, TATA
   Only cite specific paragraphs if they appear in the retrieved RAG context.
   If no RAG context is available, state the general principle without citing
   specific article numbers to avoid hallucination.
5) Provide 2-3 concrete recommended follow-up audit procedures.
6) Keep response under 250 words.
7) Write the entire response in Bahasa Indonesia professional.
""".strip()


def stream_llm(prompt: str, provider_override: str | None = None):
    provider = (provider_override or AI_PROVIDER or "").lower()
    if provider == "openrouter":
        yield from _stream_openrouter(prompt)
    elif provider in ("ollama", "local"):
        yield from _stream_ollama(prompt)
    else:
        yield f"Layanan LLM tidak tersedia. AI_PROVIDER '{provider}' tidak didukung."

def _stream_ollama(prompt: str):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    try:
        response = requests.post(f"{base_url}/api/generate", json=payload, stream=True, timeout=600)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                yield data.get("response", "")
    except Exception as exc:
        yield f" Layanan LLM tidak tersedia. Detail error: {exc}"

def _stream_openrouter(prompt: str):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your-openrouter-api-key-here":
        yield "OpenRouter API key belum diatur."
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
    }
    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                text = line.decode('utf-8')
                if text.startswith("data: "):
                    text = text[6:]
                if text.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(text)
                    if data.get("choices") and data["choices"][0].get("delta"):
                        yield data["choices"][0]["delta"].get("content", "")
                except Exception:
                    pass
    except Exception as exc:
        yield f" Layanan LLM tidak tersedia. Detail error: {exc}"

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
        "options": {"temperature": 0.2, "num_ctx": 4096},
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
    prompt = build_llm_prompt(payload, ratios, m_score, risk_status, rag_chunks)
    
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
    
    context_lines = [
        f"[{idx}] {chunk.get('content', '')}\nMetadata: {json.dumps(chunk.get('metadata', {}), ensure_ascii=True)}"
        for idx, chunk in enumerate(rag_chunks, start=1)
    ]

    prompt = f"""
You are a forensic accounting assistant. The user has provided the 'Catatan atas Laporan Keuangan' (CaLK) / Notes to Financial Statements to deepen the fraud analysis.

Previous Beneish M-Score Results:
- M-Score: {m_score}
- Risk Status: {risk_status}
- Ratios: {ratios_json}
- Raw Financial Variables (Absolute Values): {extracted_variables_json}

Retrieved Context (RAG from PSAK 115 / PSAK 72 and OJK sanction-related references):
{chr(10).join(context_lines)}

Excerpts from the uploaded CaLK document (truncated):
{text_content}

Instructions:
1) Review the provided CaLK text specifically looking for justifications, anomalies, or disclosures that explain the high-risk ratios (e.g., related party transactions, unusual revenue recognition, changes in accounting estimates). Pay special attention to the raw financial variables provided.
2) Explain how specific notes in the CaLK validate or mitigate the fraud risk signaled by the M-Score.
3) Point out any suspicious transactions (like 'Mahata' or similar unusual entities) if mentioned in the text.
4) PSAK 115 / PSAK 72 Analysis — relate your findings from the CaLK to the relevant revenue recognition steps based on the retrieved RAG context.
5) Write the entire response in professional Bahasa Indonesia.
""".strip()

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
