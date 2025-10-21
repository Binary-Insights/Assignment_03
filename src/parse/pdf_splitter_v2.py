#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF Splitter Utility — fast & predictable

- Backends:
  - PyMuPDF (fitz) preferred: compiled, very fast page-range copy
  - pypdf fallback: pure-Python (slower), but robust
  - Optional: qpdf CLI for maximum throughput
- Modes:
  * split_by_page_count (fastest and most deterministic)
  * split_by_file_size (one-pass adaptive; no inner shrink/retry loop)
- Implementation details:
  * Reuses a single open fitz.Document across all chunks
  * For PyMuPDF saves: garbage=4 and deflate=True for compact output
  * Rotating logs + JSON manifest per run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------- Optional fast backend ----------
_PYMUPDF = False
try:
    import fitz  # PyMuPDF
    _PYMUPDF = True
except Exception:
    _PYMUPDF = False

# ---------- Primary minimal backend (fallback) ----------
_PYPDF = False
try:
    from pypdf import PdfReader as _PdfReader, PdfWriter as _PdfWriter
    _PYPDF = True
except Exception:
    _PYPDF = False


@dataclass
class SplitConfig:
    pages_per_chunk: Optional[int] = None
    max_file_size_mb: Optional[float] = None
    output_base_dir: str = "data/raw"
    preserve_metadata: bool = True
    create_manifest: bool = True
    log_dir: str = "data/logs/pdf_splitting"
    log_level: str = "INFO"
    engine: str = "auto"  # auto | pymupdf | pypdf | qpdf
    qpdf_path: str = "qpdf"  # used if engine == qpdf


@dataclass
class SplitResult:
    original_file: str
    total_pages: int
    total_chunks: int
    chunk_files: List[str]
    chunk_pages: List[int]
    chunk_sizes: List[float]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class PDFSplitter:
    def __init__(self, config: SplitConfig | None = None):
        self.cfg = config or SplitConfig()
        self.log = self._setup_logging(self.cfg.log_dir, self.cfg.log_level)
        self.log.debug("Initialized PDFSplitter with config: %s", self.cfg)
        if self.cfg.engine == "auto":
            if _PYMUPDF:
                self.engine = "pymupdf"
            elif _PYPDF:
                self.engine = "pypdf"
            else:
                self.engine = "qpdf"  # last resort if available on system
        else:
            self.engine = self.cfg.engine
        self.log.info("Selected engine: %s", self.engine)

    # --- logging ---
    def _setup_logging(self, log_dir: str, level: str) -> logging.Logger:
        logger = logging.getLogger("PDFSplitter")
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

        # Console
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logger.level)
        ch.setFormatter(fmt)
        if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
            logger.addHandler(ch)

        # File (with fallback)
        try:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            log_path = log_dir_path / "pdf_splitter.log"
        except Exception:
            tmp_dir = Path("/tmp/pdf_splitter_logs")
            tmp_dir.mkdir(parents=True, exist_ok=True)
            log_path = tmp_dir / "pdf_splitter.log"

        fh = RotatingFileHandler(str(log_path), maxBytes=2_000_000, backupCount=3)
        fh.setLevel(logger.level)
        fh.setFormatter(fmt)
        if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
            logger.addHandler(fh)

        logger.info("Logging to %s", log_path)
        return logger

    # --- helpers ---
    def _open_reader(self, pdf: Path):
        if not _PYPDF:
            raise RuntimeError("pypdf not available")
        try:
            return _PdfReader(str(pdf), strict=False)  # type: ignore
        except TypeError:
            return _PdfReader(str(pdf))

    def _total_pages(self, pdf: Path, reader=None) -> int:
        if self.engine == "pymupdf" and _PYMUPDF:
            with fitz.open(str(pdf)) as d:
                return d.page_count
        if self.engine == "qpdf":
            # fall back to pypdf for counting if available
            if _PYPDF:
                r = self._open_reader(pdf)
                return len(r.pages)
            # worst-case: try PyMuPDF one-shot open
            if _PYMUPDF:
                with fitz.open(str(pdf)) as d:
                    return d.page_count
            raise RuntimeError("No backend available to count pages.")
        # pypdf path
        if reader is None:
            reader = self._open_reader(pdf)
        return len(reader.pages)

    def _extract_metadata(self, pdf: Path, reader=None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {}
        try:
            if self.engine == "pymupdf" and _PYMUPDF:
                with fitz.open(str(pdf)) as d:
                    m = d.metadata or {}
                    meta["title"] = m.get("title") or m.get("Title") or "Unknown"
                    meta["author"] = m.get("author") or m.get("Author") or ""
                    meta["subject"] = m.get("subject") or m.get("Subject") or ""
                    meta["creator"] = m.get("creator") or m.get("Creator") or ""
                    return meta
            if _PYPDF:
                if reader is None:
                    reader = self._open_reader(pdf)
                md = getattr(reader, "metadata", None) or {}
                meta["title"] = md.get("/Title", "Unknown")
                meta["author"] = md.get("/Author", "")
                meta["subject"] = md.get("/Subject", "")
                meta["creator"] = md.get("/Creator", "")
        except Exception as e:
            self.log.debug("Metadata extraction failed: %s", e)
        return meta

    def _make_output_dir(self, base: str, ticker: Optional[str]) -> Path:
        if ticker:
            out = Path(base) / ticker / "pdf" / "split_pdfs"
        else:
            out = Path(base) / "pdf" / "split_pdfs"
        out.mkdir(parents=True, exist_ok=True)
        self.log.info("Output directory: %s", out)
        return out

    def _temp_pdf_in(self, directory: Path, prefix: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(dir=str(directory), prefix=prefix, suffix=".pdf", delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        return tmp_path

    def _file_size_mb(self, p: Path) -> float:
        return p.stat().st_size / (1024 * 1024)

    # --- writing ---
    def _write_chunk_pymupdf(
        self,
        src_doc: fitz.Document,
        tmp_target: Path,
        start_page: int,
        end_page: int,
        preserve_metadata: bool,
    ) -> None:
        new = fitz.open()
        # insert_pdf uses inclusive page indices; we convert [start, end)
        new.insert_pdf(src_doc, from_page=start_page, to_page=end_page - 1)
        if preserve_metadata:
            try:
                m = src_doc.metadata or {}
                m.setdefault("producer", "PDFSplitter")
                new.set_metadata(m)
            except Exception:
                pass
        # garbage=4 is most aggressive; deflate compresses streams
        new.save(str(tmp_target), deflate=True, garbage=4)
        new.close()

    def _write_chunk_pypdf(
        self,
        source_pdf: Path,
        reader,
        tmp_target: Path,
        start_page: int,
        end_page: int,
        metadata: Dict[str, Any],
        preserve_metadata: bool,
    ) -> None:
        writer = _PdfWriter()
        for p in range(start_page, end_page):
            writer.add_page(reader.pages[p])
        if preserve_metadata and metadata:
            try:
                writer.add_metadata({
                    "/Title": metadata.get("title", "Chunk"),
                    "/Author": metadata.get("author", ""),
                    "/Subject": metadata.get("subject", ""),
                    "/Creator": "PDFSplitter",
                })
            except Exception:
                pass
        with open(tmp_target, "wb") as f:
            writer.write(f)

    def _write_chunk_qpdf(
        self,
        src: Path,
        tmp_target: Path,
        start_page: int,
        end_page: int,
    ) -> None:
        # qpdf uses 1-based inclusive ranges
        r_start = start_page + 1
        r_end = end_page
        cmd = [
            self.cfg.qpdf_path,
            str(src),
            "--pages", ".", f"{r_start}-{r_end}", "--",
            str(tmp_target),
        ]
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cp.returncode != 0:
            raise RuntimeError(f"qpdf failed: {cp.stderr.decode('utf-8', errors='ignore')}")

    # --- public: by pages ---
    def split_by_page_count(self, pdf_path: str, pages_per_chunk: int, company_ticker: Optional[str] = None) -> SplitResult:
        t0 = datetime.now()
        src = Path(pdf_path)
        self.log.info("=" * 60)
        self.log.info("Split by PAGES | file=%s | pages_per_chunk=%s", src.name, pages_per_chunk)

        if not src.exists():
            return self._fail(f"File not found: {src}", src)

        try:
            out_dir = self._make_output_dir(self.cfg.output_base_dir, company_ticker)

            reader = None
            src_doc = None
            if self.engine == "pymupdf":
                src_doc = fitz.open(str(src))
            elif self.engine == "pypdf":
                reader = self._open_reader(src)

            total_pages = self._total_pages(src, reader)
            meta = self._extract_metadata(src, reader)
            total_chunks = math.ceil(total_pages / pages_per_chunk)
            self.log.info("Total pages=%d | chunks=%d | size=%.2f MB",
                          total_pages, total_chunks, self._file_size_mb(src))

            files: List[str] = []; pages_list: List[int] = []; sizes: List[float] = []

            for i in range(total_chunks):
                s = i * pages_per_chunk
                e = min((i + 1) * pages_per_chunk, total_pages)
                tmp = self._temp_pdf_in(out_dir, prefix=f".tmp_{src.stem}_part_{i+1:03d}_")
                final = out_dir / f"{src.stem}_part_{i+1:03d}.pdf"

                if self.engine == "pymupdf":
                    self._write_chunk_pymupdf(src_doc, tmp, s, e, self.cfg.preserve_metadata)
                elif self.engine == "pypdf":
                    self._write_chunk_pypdf(src, reader, tmp, s, e, meta, self.cfg.preserve_metadata)
                else:  # qpdf
                    self._write_chunk_qpdf(src, tmp, s, e)

                shutil.move(str(tmp), str(final))
                sz = self._file_size_mb(final)
                files.append(str(final)); pages_list.append(e - s); sizes.append(sz)
                self.log.info("Chunk %03d: pages=%d | size=%.2f MB -> %s",
                              i + 1, e - s, sz, final.name)

            if src_doc: src_doc.close()
            dt = (datetime.now() - t0).total_seconds()
            return self._success(src, total_pages, files, pages_list, sizes, meta, dt)

        except Exception as e:
            self.log.exception("Split by pages failed")
            try:
                if src_doc: src_doc.close()
            except Exception:
                pass
            return self._fail(str(e), src)

    # --- public: by size (adaptive, one-pass) ---
    def split_by_file_size(self, pdf_path: str, max_file_size_mb: float, company_ticker: Optional[str] = None) -> SplitResult:
        t0 = datetime.now()
        src = Path(pdf_path)
        self.log.info("=" * 60)
        self.log.info("Split by SIZE | file=%s | target<=%.2f MB", src.name, max_file_size_mb)

        if not src.exists():
            return self._fail(f"File not found: {src}", src)

        try:
            out_dir = self._make_output_dir(self.cfg.output_base_dir, company_ticker)

            reader = None
            src_doc = None
            if self.engine == "pymupdf":
                src_doc = fitz.open(str(src))
            elif self.engine == "pypdf":
                reader = self._open_reader(src)

            total_pages = self._total_pages(src, reader)
            meta = self._extract_metadata(src, reader)

            orig_mb = self._file_size_mb(src)
            mb_per_page = max(1e-6, orig_mb / max(1, total_pages))
            safety = 0.90
            est_pages = max(1, int(max_file_size_mb / mb_per_page * safety))
            self.log.info("total_pages=%d | original_size=%.2f MB | ~MB/page=%.6f | est_pages/chunk=%d",
                          total_pages, orig_mb, mb_per_page, est_pages)

            files: List[str] = []; pages_list: List[int] = []; sizes: List[float] = []
            chunk_idx = 0
            cur = 0
            ema_mbpp = mb_per_page

            while cur < total_pages:
                start = cur
                end = min(cur + est_pages, total_pages)

                tmp = self._temp_pdf_in(out_dir, prefix=f".tmp_{src.stem}_part_{chunk_idx+1:03d}_")
                if self.engine == "pymupdf":
                    self._write_chunk_pymupdf(src_doc, tmp, start, end, self.cfg.preserve_metadata)
                elif self.engine == "pypdf":
                    self._write_chunk_pypdf(src, reader, tmp, start, end, meta, self.cfg.preserve_metadata)
                else:  # qpdf
                    self._write_chunk_qpdf(src, tmp, start, end)

                sz = self._file_size_mb(tmp)
                final = out_dir / f"{src.stem}_part_{chunk_idx + 1:03d}.pdf"
                shutil.move(str(tmp), str(final))

                # record
                files.append(str(final)); pages_list.append(end - start); sizes.append(sz)
                self.log.info("Chunk %03d: pages=%d | size=%.2f MB -> %s",
                              chunk_idx + 1, end - start, sz, final.name)

                # update estimator for next chunk
                observed_mbpp = sz / max(1, (end - start))
                ema_mbpp = 0.5 * ema_mbpp + 0.5 * observed_mbpp
                est_pages = max(1, int(max_file_size_mb / max(ema_mbpp, 1e-6) * safety))

                cur = end
                chunk_idx += 1

            if src_doc: src_doc.close()
            dt = (datetime.now() - t0).total_seconds()
            return self._success(src, total_pages, files, pages_list, sizes, meta, dt)

        except Exception as e:
            self.log.exception("Split by size failed")
            try:
                if src_doc: src_doc.close()
            except Exception:
                pass
            return self._fail(str(e), src)

    # --- manifest & result helpers ---
    def _success(
        self,
        pdf_path: Path,
        total_pages: int,
        chunk_files: List[str],
        chunk_pages: List[int],
        chunk_sizes: List[float],
        metadata: Dict[str, Any],
        processing_time: float,
    ) -> SplitResult:
        if self.cfg.create_manifest and chunk_files:
            out_dir = Path(chunk_files[0]).parent
            self._create_manifest(pdf_path, out_dir, chunk_files, chunk_pages, chunk_sizes, metadata)

        return SplitResult(
            original_file=str(pdf_path),
            total_pages=total_pages,
            total_chunks=len(chunk_files),
            chunk_files=chunk_files,
            chunk_pages=chunk_pages,
            chunk_sizes=chunk_sizes,
            metadata=metadata,
            processing_time=processing_time,
            success=True,
        )

    def _fail(self, msg: str, pdf_path: Path) -> SplitResult:
        self.log.error("Failure: %s", msg)
        return SplitResult(
            original_file=str(pdf_path),
            total_pages=0,
            total_chunks=0,
            chunk_files=[],
            chunk_pages=[],
            chunk_sizes=[],
            metadata={},
            processing_time=0.0,
            success=False,
            error_message=msg,
        )

    def _create_manifest(
        self,
        original_file: Path,
        output_dir: Path,
        chunk_files: List[str],
        chunk_pages: List[int],
        chunk_sizes: List[float],
        metadata: Dict[str, Any],
    ) -> None:
        manifest = {
            "operation": "pdf_split",
            "timestamp": datetime.now().isoformat(),
            "original_file": str(original_file),
            "original_size_mb": self._file_size_mb(original_file),
            "total_pages": sum(chunk_pages),
            "total_chunks": len(chunk_files),
            "chunks": [
                {"file": f, "pages": p, "size_mb": s}
                for f, p, s in zip(chunk_files, chunk_pages, chunk_sizes)
            ],
            "metadata": metadata,
        }
        mf = output_dir / "split_manifest.json"
        with open(mf, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        self.log.info("Manifest created: %s", mf)


# ---------- CLI ----------
def _infer_ticker(pdf_file: str, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    parts = Path(pdf_file).parts
    if len(parts) >= 3 and parts[0] == "data" and parts[1] == "raw":
        return parts[2]
    return None


def main() -> None:
    print("=" * 60)
    print("PDF Splitter Utility (Fast & Predictable)")
    print("Break large PDFs into smaller, manageable chunks")
    print("=" * 60)

    p = argparse.ArgumentParser(description="Split large PDF files into smaller chunks")
    p.add_argument("pdf_file", type=str, help="Path to PDF file to split")
    p.add_argument("--pages", type=int, default=50, help="Pages per chunk (default: 50)")
    p.add_argument("--size", type=float, default=None, help="Max file size per chunk in MB")
    p.add_argument("--ticker", type=str, default=None, help="Company ticker for output path")
    p.add_argument("--output", type=str, default="data/raw",
                   help="Base output dir (default: data/raw). Final path: {output}/{ticker}/pdf/split_pdfs/")
    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    p.add_argument("--engine", type=str, default="auto", choices=["auto", "pymupdf", "pypdf", "qpdf"],
                   help="Backend engine (default: auto)")
    p.add_argument("--qpdf-path", type=str, default="qpdf", help="qpdf binary name/path if using engine=qpdf")
    args = p.parse_args()

    cfg = SplitConfig(
        pages_per_chunk=args.pages,
        max_file_size_mb=args.size,
        output_base_dir=args.output,
        log_level=args.log_level,
        engine=args.engine,
        qpdf_path=args.qpdf_path,
    )
    splitter = PDFSplitter(cfg)
    ticker = _infer_ticker(args.pdf_file, args.ticker)

    try:
        if args.size:
            print(f"\nSplitting {args.pdf_file} with target chunk size ≤ {args.size:.2f} MB…")
            res = splitter.split_by_file_size(args.pdf_file, args.size, ticker)
        else:
            print(f"\nSplitting {args.pdf_file} into {args.pages}-page chunks…")
            res = splitter.split_by_page_count(args.pdf_file, args.pages, ticker)

        if res.success:
            print("\n✅ Split completed successfully!")
            print(f"   Original pages:  {res.total_pages}")
            print(f"   Total chunks:    {res.total_chunks}")
            print(f"   Processing time: {res.processing_time:.2f} seconds")
            print("\nChunk details:")
            for i, (f, pgs, mb) in enumerate(zip(res.chunk_files, res.chunk_pages, res.chunk_sizes), start=1):
                print(f"   {i:02d}. {Path(f).name}: {pgs} pages, {mb:.2f} MB")
        else:
            print(f"\n❌ Split failed: {res.error_message}")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Split failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()