"""
PDF Splitter Utility - Break Large PDFs into Smaller, Manageable Chunks

This module provides functionality to split large PDF files into smaller chunks based on:
- Fixed number of pages per chunk
- Maximum file size per chunk
- Custom page ranges
- Automatic optimization

Features:
- Preserves document metadata
- Maintains page order and formatting
- Generates split reports
- Handles various PDF types
- Progress tracking and logging
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

try:
    import PyPDF2
    from PyPDF2 import PdfReader, PdfWriter
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from pypdf import PdfReader as PdfReaderAlt, PdfWriter as PdfWriterAlt
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


@dataclass
class SplitConfig:
    """Configuration for PDF splitting"""
    pages_per_chunk: Optional[int] = None
    max_file_size_mb: Optional[float] = None
    output_base_dir: str = "data/raw"  # Base directory for output
    preserve_metadata: bool = True
    add_page_numbers: bool = True
    compression: bool = True
    create_manifest: bool = True


@dataclass
class SplitResult:
    """Result of a PDF split operation"""
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
    """
    Utility class to split large PDF files into smaller, manageable chunks.
    
    Supports multiple splitting strategies:
    - Fixed pages per chunk
    - Fixed file size limits
    - Custom page ranges
    """
    
    def __init__(self, config: SplitConfig = None):
        """
        Initialize PDF Splitter.
        
        Args:
            config (SplitConfig): Configuration for splitting behavior
        """
        self.config = config or SplitConfig()
        self.logger = self._setup_logging()
        self._validate_pdf_library()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('PDFSplitter')
        logger.setLevel(logging.INFO)
        
        # Create logs directory
        log_dir = Path("data/logs/pdf_splitting")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        log_file = log_dir / 'pdf_splitter.log'
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def _validate_pdf_library(self):
        """Validate that at least one PDF library is available."""
        if not PYPDF2_AVAILABLE and not PYPDF_AVAILABLE:
            raise ImportError(
                "No PDF library found. Install one of:\n"
                "  pip install PyPDF2\n"
                "  pip install pypdf"
            )
        
        available = []
        if PYPDF2_AVAILABLE:
            available.append("PyPDF2")
        if PYPDF_AVAILABLE:
            available.append("pypdf")
        
        self.logger.info(f"PDF libraries available: {', '.join(available)}")
    
    def _get_pdf_reader(self, pdf_path: Path):
        """Get appropriate PDF reader based on available libraries."""
        if PYPDF2_AVAILABLE:
            return PdfReader(str(pdf_path))
        else:
            return PdfReaderAlt(str(pdf_path))
    
    def _get_pdf_writer(self):
        """Get appropriate PDF writer based on available libraries."""
        if PYPDF2_AVAILABLE:
            return PdfWriter()
        else:
            return PdfWriterAlt()
    
    def _get_file_size_mb(self, file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)
    
    def split_by_page_count(
        self,
        pdf_path: str,
        pages_per_chunk: int,
        company_ticker: Optional[str] = None
    ) -> SplitResult:
        """
        Split PDF into chunks with fixed number of pages.
        
        Args:
            pdf_path (str): Path to PDF file to split
            pages_per_chunk (int): Number of pages per chunk
            company_ticker (str): Company ticker for organization
            
        Returns:
            SplitResult: Information about split operation
        """
        pdf_path = Path(pdf_path)
        start_time = datetime.now()
        
        self.logger.info("="*60)
        self.logger.info(f"Starting PDF split by page count: {pdf_path.name}")
        self.logger.info(f"Pages per chunk: {pages_per_chunk}")
        
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return SplitResult(
                original_file=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                chunk_files=[],
                chunk_pages=[],
                chunk_sizes=[],
                metadata={},
                processing_time=0,
                success=False,
                error_message=f"File not found: {pdf_path}"
            )
        
        try:
            # Read the PDF
            self.logger.info("Reading PDF file...")
            reader = self._get_pdf_reader(pdf_path)
            total_pages = len(reader.pages)
            
            self.logger.info(f"Total pages in PDF: {total_pages}")
            self.logger.info(f"File size: {self._get_file_size_mb(pdf_path):.2f} MB")
            
            # Extract metadata
            metadata = self._extract_metadata(reader)
            
            # Create output directory
            # Path: data/raw/{company_ticker}/pdf/split_pdfs/{pdf_name}/
            if company_ticker:
                output_dir = (
                    Path(self.config.output_base_dir) / 
                    company_ticker / 
                    "pdf" / 
                    "split_pdfs" 
                )
            else:
                # Fallback if no ticker provided - use pdf/split_pdfs structure
                output_dir = (
                    Path(self.config.output_base_dir) / 
                    "pdf" / 
                    "split_pdfs" 
                )
            
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir}")
            
            # Calculate chunks
            total_chunks = (total_pages + pages_per_chunk - 1) // pages_per_chunk
            self.logger.info(f"Total chunks to create: {total_chunks}")
            
            chunk_files = []
            chunk_pages = []
            chunk_sizes = []
            
            # Create chunks
            for chunk_idx in range(total_chunks):
                start_page = chunk_idx * pages_per_chunk
                end_page = min((chunk_idx + 1) * pages_per_chunk, total_pages)
                pages_in_chunk = end_page - start_page
                
                self.logger.info(f"Creating chunk {chunk_idx + 1}/{total_chunks}...")
                self.logger.debug(f"  Pages {start_page} to {end_page} ({pages_in_chunk} pages)")
                
                # Create chunk file
                chunk_file = output_dir / f"{pdf_path.stem}_part_{chunk_idx + 1:03d}.pdf"
                self._write_pdf_chunk(
                    reader, chunk_file, start_page, end_page, metadata
                )
                
                chunk_size = self._get_file_size_mb(chunk_file)
                chunk_files.append(str(chunk_file))
                chunk_pages.append(pages_in_chunk)
                chunk_sizes.append(chunk_size)
                
                self.logger.debug(f"  Chunk file: {chunk_file.name} ({chunk_size:.2f} MB)")
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create manifest
            if self.config.create_manifest:
                self._create_manifest(
                    pdf_path, output_dir, chunk_files, chunk_pages, chunk_sizes, metadata
                )
            
            result = SplitResult(
                original_file=str(pdf_path),
                total_pages=total_pages,
                total_chunks=total_chunks,
                chunk_files=chunk_files,
                chunk_pages=chunk_pages,
                chunk_sizes=chunk_sizes,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )
            
            self.logger.info(f"PDF split completed successfully")
            self.logger.info(f"Created {total_chunks} chunk(s)")
            self.logger.info(f"Processing time: {processing_time:.2f} seconds")
            self.logger.info("="*60)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error splitting PDF: {e}", exc_info=True)
            return SplitResult(
                original_file=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                chunk_files=[],
                chunk_pages=[],
                chunk_sizes=[],
                metadata={},
                processing_time=0,
                success=False,
                error_message=str(e)
            )
    
    def split_by_file_size(
        self,
        pdf_path: str,
        max_file_size_mb: float,
        company_ticker: Optional[str] = None
    ) -> SplitResult:
        """
        Split PDF into chunks with maximum file size.
        
        Args:
            pdf_path (str): Path to PDF file to split
            max_file_size_mb (float): Maximum file size per chunk in MB
            company_ticker (str): Company ticker for organization
            
        Returns:
            SplitResult: Information about split operation
        """
        pdf_path = Path(pdf_path)
        start_time = datetime.now()
        
        self.logger.info("="*60)
        self.logger.info(f"Starting PDF split by file size: {pdf_path.name}")
        self.logger.info(f"Max file size: {max_file_size_mb:.2f} MB")
        
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return SplitResult(
                original_file=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                chunk_files=[],
                chunk_pages=[],
                chunk_sizes=[],
                metadata={},
                processing_time=0,
                success=False,
                error_message=f"File not found: {pdf_path}"
            )
        
        try:
            # Read the PDF
            self.logger.info("Reading PDF file...")
            reader = self._get_pdf_reader(pdf_path)
            total_pages = len(reader.pages)
            
            original_size = self._get_file_size_mb(pdf_path)
            self.logger.info(f"Total pages in PDF: {total_pages}")
            self.logger.info(f"Original file size: {original_size:.2f} MB")
            
            # Extract metadata
            metadata = self._extract_metadata(reader)
            
            # Create output directory
            # Path: data/raw/{company_ticker}/pdf/split_pdfs/{pdf_name}/
            if company_ticker:
                output_dir = (
                    Path(self.config.output_base_dir) / 
                    company_ticker / 
                    "pdf" / 
                    "split_pdfs" 
                )
            else:
                # Fallback if no ticker provided - use pdf/split_pdfs structure
                output_dir = (
                    Path(self.config.output_base_dir) / 
                    "pdf" / 
                    "split_pdfs" 
                )
            
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir}")
            
            # Estimate pages per chunk based on original size
            estimated_pages_per_chunk = max(
                1,
                int(total_pages * max_file_size_mb / original_size)
            )
            self.logger.info(f"Estimated pages per chunk: {estimated_pages_per_chunk}")
            
            chunk_files = []
            chunk_pages = []
            chunk_sizes = []
            chunk_idx = 0
            
            # Create chunks
            current_page = 0
            while current_page < total_pages:
                # Start with estimated pages, adjust if needed
                end_page = min(current_page + estimated_pages_per_chunk, total_pages)
                
                # Refine chunk size to meet file size requirement
                while end_page > current_page:
                    chunk_file_temp = output_dir / f"temp_{chunk_idx}.pdf"
                    self._write_pdf_chunk(
                        reader, chunk_file_temp, current_page, end_page, metadata
                    )
                    
                    chunk_size = self._get_file_size_mb(chunk_file_temp)
                    
                    if chunk_size <= max_file_size_mb or end_page - current_page == 1:
                        # Size is acceptable or we're at single page
                        chunk_file = output_dir / f"{pdf_path.stem}_part_{chunk_idx + 1:03d}.pdf"
                        chunk_file_temp.rename(chunk_file)
                        
                        pages_in_chunk = end_page - current_page
                        chunk_files.append(str(chunk_file))
                        chunk_pages.append(pages_in_chunk)
                        chunk_sizes.append(chunk_size)
                        
                        self.logger.info(
                            f"Chunk {chunk_idx + 1}: {pages_in_chunk} pages, "
                            f"{chunk_size:.2f} MB"
                        )
                        
                        current_page = end_page
                        chunk_idx += 1
                        break
                    else:
                        # Size too large, reduce end page
                        end_page -= max(1, (end_page - current_page) // 2)
                        chunk_file_temp.unlink()
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create manifest
            if self.config.create_manifest:
                self._create_manifest(
                    pdf_path, output_dir, chunk_files, chunk_pages, chunk_sizes, metadata
                )
            
            result = SplitResult(
                original_file=str(pdf_path),
                total_pages=total_pages,
                total_chunks=len(chunk_files),
                chunk_files=chunk_files,
                chunk_pages=chunk_pages,
                chunk_sizes=chunk_sizes,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )
            
            self.logger.info(f"PDF split completed successfully")
            self.logger.info(f"Created {len(chunk_files)} chunk(s)")
            self.logger.info(f"Processing time: {processing_time:.2f} seconds")
            self.logger.info("="*60)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error splitting PDF: {e}", exc_info=True)
            return SplitResult(
                original_file=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                chunk_files=[],
                chunk_pages=[],
                chunk_sizes=[],
                metadata={},
                processing_time=0,
                success=False,
                error_message=str(e)
            )
    
    def split_custom_ranges(
        self,
        pdf_path: str,
        page_ranges: List[Tuple[int, int]],
        company_ticker: Optional[str] = None
    ) -> SplitResult:
        """
        Split PDF into custom page ranges.
        
        Args:
            pdf_path (str): Path to PDF file to split
            page_ranges (List[Tuple[int, int]]): List of (start, end) page tuples (0-indexed)
            company_ticker (str): Company ticker for organization
            
        Returns:
            SplitResult: Information about split operation
        """
        pdf_path = Path(pdf_path)
        start_time = datetime.now()
        
        self.logger.info("="*60)
        self.logger.info(f"Starting PDF split with custom ranges: {pdf_path.name}")
        self.logger.info(f"Number of ranges: {len(page_ranges)}")
        
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return SplitResult(
                original_file=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                chunk_files=[],
                chunk_pages=[],
                chunk_sizes=[],
                metadata={},
                processing_time=0,
                success=False,
                error_message=f"File not found: {pdf_path}"
            )
        
        try:
            # Read the PDF
            self.logger.info("Reading PDF file...")
            reader = self._get_pdf_reader(pdf_path)
            total_pages = len(reader.pages)
            
            self.logger.info(f"Total pages in PDF: {total_pages}")
            
            # Validate ranges
            for i, (start, end) in enumerate(page_ranges):
                if start < 0 or end > total_pages or start >= end:
                    raise ValueError(
                        f"Invalid range at index {i}: ({start}, {end}). "
                        f"Valid range: 0 to {total_pages}"
                    )
            
            # Extract metadata
            metadata = self._extract_metadata(reader)
            
            # Create output directory
            # Path: data/raw/{company_ticker}/pdf/split_pdfs/{pdf_name}/
            if company_ticker:
                output_dir = (
                    Path(self.config.output_base_dir) / 
                    company_ticker / 
                    "pdf" / 
                    "split_pdfs" 
                )
            else:
                # Fallback if no ticker provided - use pdf/split_pdfs structure
                output_dir = (
                    Path(self.config.output_base_dir) / 
                    "pdf" / 
                    "split_pdfs" 
                )
            
            output_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Output directory: {output_dir}")
            
            chunk_files = []
            chunk_pages = []
            chunk_sizes = []
            
            # Create chunks for each range
            for range_idx, (start_page, end_page) in enumerate(page_ranges):
                self.logger.info(f"Creating chunk {range_idx + 1}/{len(page_ranges)}...")
                self.logger.debug(f"  Pages {start_page} to {end_page}")
                
                chunk_file = output_dir / f"{pdf_path.stem}_range_{range_idx + 1:03d}.pdf"
                self._write_pdf_chunk(
                    reader, chunk_file, start_page, end_page, metadata
                )
                
                chunk_size = self._get_file_size_mb(chunk_file)
                pages_in_chunk = end_page - start_page
                
                chunk_files.append(str(chunk_file))
                chunk_pages.append(pages_in_chunk)
                chunk_sizes.append(chunk_size)
                
                self.logger.debug(f"  Chunk file: {chunk_file.name} ({chunk_size:.2f} MB)")
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create manifest
            if self.config.create_manifest:
                self._create_manifest(
                    pdf_path, output_dir, chunk_files, chunk_pages, chunk_sizes, metadata
                )
            
            result = SplitResult(
                original_file=str(pdf_path),
                total_pages=total_pages,
                total_chunks=len(chunk_files),
                chunk_files=chunk_files,
                chunk_pages=chunk_pages,
                chunk_sizes=chunk_sizes,
                metadata=metadata,
                processing_time=processing_time,
                success=True
            )
            
            self.logger.info(f"PDF split with custom ranges completed successfully")
            self.logger.info(f"Created {len(chunk_files)} chunk(s)")
            self.logger.info(f"Processing time: {processing_time:.2f} seconds")
            self.logger.info("="*60)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error splitting PDF: {e}", exc_info=True)
            return SplitResult(
                original_file=str(pdf_path),
                total_pages=0,
                total_chunks=0,
                chunk_files=[],
                chunk_pages=[],
                chunk_sizes=[],
                metadata={},
                processing_time=0,
                success=False,
                error_message=str(e)
            )
    
    def _write_pdf_chunk(
        self,
        reader,
        output_file: Path,
        start_page: int,
        end_page: int,
        metadata: Dict[str, Any]
    ):
        """Write a chunk of pages to a new PDF file."""
        writer = self._get_pdf_writer()
        
        # Copy pages
        for page_idx in range(start_page, end_page):
            writer.add_page(reader.pages[page_idx])
        
        # Add metadata if preserved
        if self.config.preserve_metadata and metadata:
            try:
                writer.add_metadata({
                    '/Title': metadata.get('title', 'Chunk'),
                    '/Author': metadata.get('author', ''),
                    '/Subject': metadata.get('subject', ''),
                    '/Creator': 'PDFSplitter'
                })
            except Exception as e:
                self.logger.warning(f"Failed to add metadata: {e}")
        
        # Write file
        with open(output_file, 'wb') as f:
            writer.write(f)
    
    def _extract_metadata(self, reader) -> Dict[str, Any]:
        """Extract metadata from PDF."""
        metadata = {}
        try:
            if hasattr(reader, 'metadata'):
                reader_metadata = reader.metadata
                if reader_metadata:
                    metadata['title'] = reader_metadata.get('/Title', 'Unknown')
                    metadata['author'] = reader_metadata.get('/Author', '')
                    metadata['subject'] = reader_metadata.get('/Subject', '')
                    metadata['creator'] = reader_metadata.get('/Creator', '')
        except Exception as e:
            self.logger.warning(f"Failed to extract metadata: {e}")
        
        return metadata
    
    def _create_manifest(
        self,
        original_file: Path,
        output_dir: Path,
        chunk_files: List[str],
        chunk_pages: List[int],
        chunk_sizes: List[float],
        metadata: Dict[str, Any]
    ):
        """Create a manifest file documenting the split operation."""
        manifest = {
            'operation': 'pdf_split',
            'timestamp': datetime.now().isoformat(),
            'original_file': str(original_file),
            'original_size_mb': self._get_file_size_mb(original_file),
            'total_pages': sum(chunk_pages),
            'total_chunks': len(chunk_files),
            'chunks': [
                {
                    'file': chunk_file,
                    'pages': pages,
                    'size_mb': size
                }
                for chunk_file, pages, size in zip(chunk_files, chunk_pages, chunk_sizes)
            ],
            'metadata': metadata
        }
        
        manifest_file = output_dir / 'split_manifest.json'
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Manifest created: {manifest_file}")


def main():
    """Main function to demonstrate PDF splitting."""
    import argparse
    import sys
    
    print("="*60)
    print("PDF Splitter Utility")
    print("Break large PDFs into smaller, manageable chunks")
    print("="*60)
    
    parser = argparse.ArgumentParser(
        description="Split large PDF files into smaller chunks"
    )
    parser.add_argument("pdf_file", type=str, help="Path to PDF file to split")
    parser.add_argument(
        "--pages",
        type=int,
        default=50,
        help="Number of pages per chunk (default: 50)"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=None,
        help="Maximum file size per chunk in MB (optional)"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Company ticker for organization"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Base output directory (default: data/raw). Final path: {output}/{ticker}/pdf/split_pdfs/"
    )
    
    args = parser.parse_args()
    
    # Extract company ticker from file path if not provided
    ticker = args.ticker
    if not ticker:
        # Try to extract ticker from path like data/raw/FINTBX/pdf/fintbx.pdf
        pdf_parts = Path(args.pdf_file).parts
        if len(pdf_parts) >= 3 and pdf_parts[0] == "data" and pdf_parts[1] == "raw":
            ticker = pdf_parts[2]  # Extract FINTBX from the path
    
    # Create config
    config = SplitConfig(
        pages_per_chunk=args.pages,
        max_file_size_mb=args.size,
        output_base_dir=args.output
    )
    
    # Create splitter
    splitter = PDFSplitter(config)
    
    # Split PDF
    if args.size:
        print(f"\nSplitting {args.pdf_file} by file size ({args.size} MB max)...")
        result = splitter.split_by_file_size(
            args.pdf_file,
            args.size,
            ticker
        )
    else:
        print(f"\nSplitting {args.pdf_file} into {args.pages}-page chunks...")
        result = splitter.split_by_page_count(
            args.pdf_file,
            args.pages,
            ticker
        )
    
    # Print results
    if result.success:
        print(f"\n✅ Split completed successfully!")
        print(f"   Original pages: {result.total_pages}")
        print(f"   Total chunks: {result.total_chunks}")
        print(f"   Processing time: {result.processing_time:.2f} seconds")
        print(f"\nChunk details:")
        for i, (chunk_file, pages, size) in enumerate(
            zip(result.chunk_files, result.chunk_pages, result.chunk_sizes)
        ):
            print(f"   {i + 1}. {Path(chunk_file).name}: {pages} pages, {size:.2f} MB")
    else:
        print(f"\n❌ Split failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
