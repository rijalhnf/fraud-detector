import json
from typing import Any

# for /api/analyze
def build_llm_prompt(
    company_name: str,
    fiscal_year: str,
    ratios: dict[str, float],
    m_score: float,
    risk_status: str,
    rag_chunks: list[dict[str, Any]],
) -> str:
    benchmark_reference = {
        "DSRI": {"threshold": 1.465, "insight": "Indikasi channel stuffing atau pengakuan pendapatan terlalu dini.", "note": "Dihitung dari piutang usaha (trade receivables) saja."},
        "GMI": {"threshold": 1.193, "insight": "Margin memburuk dapat mendorong manajemen menyamarkan kerugian.", "note": "Jika kedua margin negatif (perusahaan rugi), GMI < 1 secara mekanis namun margin ekonomis memburuk."},
        "AQI": {"threshold": 1.254, "insight": "Kualitas aset menurun akibat deferral atau kapitalisasi biaya agresif."},
        "SGI": {"threshold": 1.607, "insight": "Pertumbuhan tinggi meningkatkan tekanan untuk mempertahankan ekspektasi pasar."},
        "DEPI": {"threshold": 1.077, "insight": "Perlambatan penyusutan dapat menaikkan laba secara artifisial (perpanjangan umur aset).", "note": "Menggunakan beban penyusutan aktual dari CLK."},
        "SGAI": {"threshold": 1.041, "insight": "Efisiensi beban SG&A melemah terhadap pendapatan.", "note": "Dihitung dari full SG&A = beban administrasi + beban penjualan."},
        "LVGI": {"threshold": 1.111, "insight": "Leverage naik meningkatkan risiko manipulasi untuk memenuhi covenant utang.", "note": "Menggunakan utang berbunga saja."},
        "TATA": {"threshold": 0.031, "insight": "Akrual tinggi relatif terhadap arus kas mengindikasikan kualitas laba rendah.", "note": "Metode income statement: (Laba Usaha - Arus Kas Operasi) / Total Aset."},
    }

    triggered_flags = {}
    for r_name, r_val in ratios.items():
        ref = benchmark_reference.get(r_name)
        if ref and r_val > ref["threshold"]:
            triggered_flags[r_name] = {
                "value": round(r_val, 4),
                "threshold": ref["threshold"],
                "insight": ref["insight"],
                "note": ref.get("note", "")
            }

    # Only include the actual text content to save tokens (no metadata JSON dump)
    context_lines = [
        f"[{idx}] {chunk.get('content', '').strip()}"
        for idx, chunk in enumerate(rag_chunks, start=1)
    ]

    company = company_name or "Unknown Company"
    year = fiscal_year or "Unknown Fiscal Year"

    return f"""
You are a forensic accounting assistant for an AI Fraud Early Warning System.
Analyze the company's fraud risk using Beneish M-Score results and regulatory/accounting context.

Company: {company}
Fiscal Year: {year}

All Calculated Ratios:
{json.dumps({k: round(v, 4) for k, v in ratios.items()}, indent=2)}

Final M-Score: {m_score:.4f}
Risk Status: {risk_status} (Threshold: > -1.78 High Risk, > -2.22 Medium Risk)

Triggered Red Flags (Exceeded Thresholds):
{json.dumps(triggered_flags, indent=2) if triggered_flags else "None. All ratios are within safe limits."}

Retrieved Context (RAG from PSAK 115/72):
{chr(10).join(context_lines) if context_lines else "No specific regulatory context retrieved."}

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
6) Provide a comprehensive and highly detailed forensic analysis (at least 400 words).
7) Write the entire response in Bahasa Indonesia professional.
""".strip()


# for /api/analyze-calk
def build_calk_prompt(
    m_score: float,
    risk_status: str,
    ratios_json: str,
    extracted_variables_json: str,
    rag_chunks: list[dict[str, Any]],
    text_content: str,
) -> str:
    context_lines = [
        f"[{idx}] {chunk.get('content', '').strip()}"
        for idx, chunk in enumerate(rag_chunks, start=1)
    ]

    return f"""
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
