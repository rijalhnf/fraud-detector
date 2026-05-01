from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
from typing import Iterable

import chromadb


def extract_pdf_text(pdf_path: Path) -> str:
    text_parts: list[str] = []

    try:
        import fitz  # type: ignore

        with fitz.open(pdf_path) as doc:
            for page in doc:
                text_parts.append(page.get_text("text") or "")
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    try:
        import pdfplumber  # type: ignore

        with pdf_path.open("rb") as f:
            pdf_bytes = f.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts).strip()
    except Exception:
        return ""


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(file_path)
    if suffix in {".txt", ".md"}:
        return file_path.read_text(encoding="utf-8", errors="ignore")
    return ""


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> Iterable[str]:
    # Clean up excessive whitespace but preserve newlines
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(line for line in lines if line)
    
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    
    while start < n:
        end = min(start + chunk_size, n)
        
        # Try to find a natural break point (newline or period) within the overlap zone
        if end < n:
            # Look back up to 'overlap' characters to find a newline
            newline_pos = cleaned.rfind("\n", max(start, end - overlap), end)
            if newline_pos != -1:
                end = newline_pos + 1
            else:
                # Fallback to period if no newline
                period_pos = cleaned.rfind(". ", max(start, end - overlap), end)
                if period_pos != -1:
                    end = period_pos + 2

        chunks.append(cleaned[start:end].strip())
        
        if end >= n:
            break
            
        start = end
        
    return chunks


def infer_doc_kind(file_name: str) -> str:
    name = file_name.lower()
    if "psak" in name and "72" in name:
        return "psak_72"
    if "ojk" in name and ("sanction" in name or "letter" in name or "surat" in name):
        return "ojk_sanction_letter"
    return "other"


def stable_chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
    return f"{source}-{index}-{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PSAK/OJK docs into ChromaDB")
    parser.add_argument("--input-dir", default="/app/knowledge", help="Directory containing PDF/TXT/MD files")
    parser.add_argument("--db-path", default=os.getenv("CHROMA_DB_PATH", "./chroma_db"), help="Persistent ChromaDB path")
    parser.add_argument(
        "--collection",
        default=os.getenv("CHROMA_COLLECTION_NAME", "fraud_knowledge"),
        help="Chroma collection name",
    )
    parser.add_argument("--reset", action="store_true", help="Delete and recreate collection before ingest")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}
    )
    if not files:
        raise SystemExit(f"No supported files found in: {input_dir}")

    client = chromadb.PersistentClient(path=args.db_path)

    if args.reset:
        try:
            client.delete_collection(name=args.collection)
        except Exception:
            pass

    collection = client.get_or_create_collection(name=args.collection)

    all_ids: list[str] = []
    all_docs: list[str] = []
    all_meta: list[dict[str, str]] = []

    for file_path in files:
        text = extract_text(file_path)
        if not text.strip():
            continue

        rel_source = str(file_path.relative_to(input_dir))
        doc_kind = infer_doc_kind(file_path.name)

        for idx, chunk in enumerate(chunk_text(text), start=1):
            chunk_id = stable_chunk_id(rel_source.replace("/", "_"), idx, chunk)
            all_ids.append(chunk_id)
            all_docs.append(chunk)
            all_meta.append(
                {
                    "source": rel_source,
                    "doc_kind": doc_kind,
                    "chunk_index": str(idx),
                }
            )

    if not all_docs:
        raise SystemExit("No text chunks extracted from input files.")

    collection.upsert(ids=all_ids, documents=all_docs, metadatas=all_meta)
    print(f"Ingested {len(all_docs)} chunks from {len(files)} files into '{args.collection}'.")


if __name__ == "__main__":
    main()
