from typing import Any
from pydantic import BaseModel, ConfigDict, Field

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
