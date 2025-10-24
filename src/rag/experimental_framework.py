import os
import sys
import json
import time
import math
from pathlib import Path
from statistics import mean, pstdev  # population std dev (old summary looked like population)
from typing import Dict, List, Tuple, Any

from tqdm import tqdm
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)


# -------------------------------
#  Token counting (with fallback)
# -------------------------------
def _make_token_counter():
    """Return a function(text) -> token_count. Falls back to len(text) if tiktoken not available."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def count_tokens(text: str) -> int:
            return len(enc.encode(text or ""))

        return count_tokens
    except Exception:
        def count_chars(text: str) -> int:
            return len(text or "")
        return count_chars

count_tokens = _make_token_counter()


# -------------------------------
#  I/O helpers
# -------------------------------
def read_text_from_part(part_path: Path) -> str:
    """Read structured_content.txt or full_document.txt from a part folder."""
    structured_path = part_path / "text" / "structured_content.txt"
    full_path = part_path / "text" / "full_document.txt"

    if structured_path.exists():
        return structured_path.read_text(encoding="utf-8", errors="ignore")
    if full_path.exists():
        return full_path.read_text(encoding="utf-8", errors="ignore")
    print(f"⚠️  No text files found in {part_path}")
    return ""


def collect_all_texts_and_first_source(base_path: Path, ticker: str) -> Tuple[str, str]:
    """
    Reads and concatenates text from all {ticker_lower}_part_* folders.
    Returns (combined_text, first_source_path_for_summary).
    """
    pattern = f"{ticker.lower()}_part_*"
    part_folders = sorted(base_path.glob(pattern))
    texts: List[str] = []
    first_source = ""

    for i, folder in enumerate(tqdm(part_folders, desc="Loading parts")):
        txt = read_text_from_part(folder)
        if txt.strip():
            texts.append(txt)
            if not first_source:  # capture first existing file path for the 'document' field
                # Prefer structured_content.txt if present, else full_document.txt
                s1 = folder / "text" / "structured_content.txt"
                s2 = folder / "text" / "full_document.txt"
                first = s1 if s1.exists() else s2
                first_source = str(first)

    return ("\n\n".join(texts), first_source or str(base_path))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


# -------------------------------
#  Chunking strategies
# -------------------------------
def chunk_recursive(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    return splitter.create_documents([text])


def chunk_markdownheader(text: str):
    headers = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    # Returns a list of Document-like objects with .page_content/.metadata
    return splitter.split_text(text)


def chunk_codeaware(text: str):
    """Simple code-aware: keep fenced code blocks intact; split non-code recursively."""
    code_blocks: List[str] = []
    non_code_lines: List[str] = []
    inside_code = False
    buffer: List[str] = []

    for line in text.splitlines(keepends=True):
        if line.strip().startswith("```"):
            inside_code = not inside_code
            if not inside_code:
                code_blocks.append("".join(buffer))
                buffer = []
            continue
        if inside_code:
            buffer.append(line)
        else:
            non_code_lines.append(line)

    # If we ended inside a fence, flush anyway
    if buffer:
        code_blocks.append("".join(buffer))

    non_code_text = "".join(non_code_lines)
    rc_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = rc_splitter.create_documents([non_code_text])

    # Append code chunks as separate documents with metadata.type="code"
    for block in code_blocks:
        docs.append({"page_content": block, "metadata": {"type": "code"}})

    return docs


def chunk_hybrid(text: str):
    """Combine recursive + markdown-aware outputs."""
    return chunk_recursive(text) + chunk_markdownheader(text)


# -------------------------------
#  Old-style chunk JSON schema
# -------------------------------
STRATEGY_FILE_KEYS = {
    "Recursive": "chunks_recursive",
    "Hybrid": "chunks_hybrid",
    "MarkdownHeader": "chunks_markdownheader",
    "CodeAware": "chunks_codeaware",
}

def to_old_style_chunks(
    ticker: str,
    strategy_name: str,
    chunks: List[Any],
) -> List[Dict[str, Any]]:
    """
    Convert list of doc/chunk objects to the old schema:
      {
        "text": "...",
        "chunk_id": "chunk_000000",
        "chunk_strategy": "<Strategy>",
        "doc_id": "<ticker>_all_parts",
        "doc_title": "...",
        "content_type": "general" | "code" | ...,
        "technical_level": "beginner",
        "keywords": [],
        "has_code": bool,
        "has_math": false,
        "page_number": <int>,
        "position_in_doc": <float>
      }
    """
    out: List[Dict[str, Any]] = []
    n = max(1, len(chunks))
    for idx, c in enumerate(chunks):
        if isinstance(c, dict):
            text = c.get("page_content") or c.get("text", "")
            metadata = c.get("metadata", {}) or {}
        else:
            text = getattr(c, "page_content", "")
            metadata = getattr(c, "metadata", {}) or {}

        content_type = metadata.get("type", "general")
        page_no = metadata.get("page", 1)

        out.append({
            "text": text,
            "chunk_id": f"chunk_{idx:06d}",
            "chunk_strategy": strategy_name,
            "doc_id": f"{ticker.lower()}_all_parts",
            "doc_title": "Financial Toolbox™ User's Guide",
            "content_type": content_type,
            "technical_level": "beginner",
            "keywords": [],
            "has_code": (content_type == "code"),
            "has_math": False,
            "page_number": page_no,
            "position_in_doc": round(idx / n, 3),
        })
    return out


# -------------------------------
#  Stats for summary.json (old style)
# -------------------------------
def compute_summary_stats(strategy_name: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build the detailed metrics section like the old summary:
      - chunk_count
      - total_tokens (tokens if tiktoken available; else characters)
      - mean/min/max/std_dev (based on token counts / char counts)
      - execution_time_ms handled by caller (we pass it in)
    """
    sizes = [count_tokens(c.get("text", "")) for c in chunks]
    chunk_count = len(sizes)
    total_tokens = int(sum(sizes))
    if chunk_count == 0:
        return {
            "chunk_count": 0,
            "total_tokens": 0,
            "mean_chunk_size": 0.0,
            "min_chunk_size": 0,
            "max_chunk_size": 0,
            "std_dev": 0.0,
            "execution_time_ms": 0.0,
        }

    mu = mean(sizes)
    sigma = pstdev(sizes) if chunk_count > 1 else 0.0
    return {
        "chunk_count": chunk_count,
        "total_tokens": total_tokens,
        "mean_chunk_size": round(mu, 2),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
        "std_dev": round(sigma, 2),
        # execution_time_ms added by caller
    }


# -------------------------------
#  Main pipeline
# -------------------------------
def run_experiment(ticker: str):
    base_dir = Path(f"data/parsed/{ticker}")
    output_dir = Path("data/rag_experiments")
    ensure_dir(output_dir)

    print(f"\n🔍 Collecting parsed text parts for ticker: {ticker}")
    combined_text, first_source = collect_all_texts_and_first_source(base_dir, ticker)
    doc_size_chars = len(combined_text)
    print(f"✅ Combined text length: {doc_size_chars:,} characters\n")

    strategies = {
        "Recursive": chunk_recursive,
        "Hybrid": chunk_hybrid,
        "MarkdownHeader": chunk_markdownheader,
        "CodeAware": chunk_codeaware,
    }

    # For summary.json in old style
    detailed_results: Dict[str, Dict[str, Any]] = {}

    for strat_name, splitter_fn in strategies.items():
        print(f"→ Running {strat_name} ...")
        t0 = time.perf_counter()
        raw_chunks = splitter_fn(combined_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Convert to old-style chunk JSON schema
        old_chunks = to_old_style_chunks(ticker, strat_name, raw_chunks)

        # Save chunk file to the expected filename
        file_key = STRATEGY_FILE_KEYS[strat_name]  # e.g., "chunks_recursive"
        out_path = output_dir / f"{file_key}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(old_chunks, f, indent=2)
        print(f"  ✓ Saved {len(old_chunks)} chunks to {out_path}")

        # Collect metrics for the old-style detailed summary
        stats = compute_summary_stats(strat_name, old_chunks)
        stats["execution_time_ms"] = round(elapsed_ms, 1)
        detailed_results[strat_name] = stats

    # Build old-style summary.json payload
    summary_payload = {
        "document": first_source,
        "document_size_chars": doc_size_chars,
        "strategies_tested": len(strategies),
        "results": detailed_results,
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n📊 Summary (old style):")
    print(json.dumps(summary_payload, indent=2))
    print(f"\n✅ All outputs written to {output_dir.resolve()}\n")


# -------------------------------
#  Entry Point
# -------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experimental_framework.py <TICKER>")
        sys.exit(1)

    ticker_arg = sys.argv[1].strip().upper()
    run_experiment(ticker_arg)