"""
Docling-based Advanced PDF Understanding and Content Extraction

This module uses IBM's Docling library for sophisticated PDF analysis including:
- Reading order analysis
- Advanced table extraction
- Formula recognition
- Unified DoclingDocument format
- Export to Markdown and JSON

Docling provides state-of-the-art document understanding capabilities
that go beyond traditional text extraction methods.
"""

try:
    from docling.document_converter import DocumentConverter
    # , PdfFormatOption
    # from docling.datamodel.base_models import InputFormat
    # from docling.datamodel.pipeline_options import PdfPipelineOptions
    # from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import base64


class DoclingExtractor:
    """
    Advanced PDF content extraction using IBM's Docling library.
    
    This class leverages Docling's sophisticated document understanding
    capabilities for comprehensive PDF analysis and content extraction.
    """
    
    def __init__(self, output_dir="data/parsed/docling", company_ticker=None):
        """
        Initialize the Docling-based extractor.
        
        Args:
            output_dir (str): Directory to save extracted content
            company_ticker (str): Company ticker for logging directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.company_ticker = company_ticker
        
        # Check if Docling is available
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling\n"
                "Note: Docling requires Python 3.9+ and may need additional system dependencies."
            )
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize document converter with advanced options
        self.converter = self._initialize_converter()
        
        # Extraction statistics
        self.stats = {
            'total_pages': 0,
            'total_documents': 0,
            'reading_order_elements': 0,
            'tables_extracted': 0,
            'formulas_detected': 0,
            'figures_detected': 0,
            'processing_time': 0,
            'export_formats': []
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logger = logging.getLogger('DoclingExtractor')
        logger.setLevel(logging.INFO)
        
        # Create logs directory based on company ticker
        if self.company_ticker:
            log_dir = Path(f"data/logs/{self.company_ticker}")
        else:
            log_dir = Path("data/logs/docling")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        log_file = log_dir / 'docling_extraction.log'
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
    
    def _initialize_converter(self):
        """Initialize Docling document converter with basic configuration for v1.20.0."""
        try:
            # Simple initialization compatible with Docling v1.20.0
            converter = DocumentConverter()
            
            # Print model information for debugging
            print("\n=== Docling Model Information ===")
            if hasattr(converter, 'pipeline_options'):
                print(f"Pipeline Options: {converter.pipeline_options}")
            if hasattr(converter, 'format_options'):
                print(f"Format Options: {converter.format_options}")
            if hasattr(converter, 'artifact_path'):
                print(f"Artifact Path: {converter.artifact_path}")
            
            # Try to find model configuration
            for attr in dir(converter):
                if 'model' in attr.lower() or 'config' in attr.lower():
                    try:
                        value = getattr(converter, attr)
                        if not callable(value) and str(value) != '<built-in method __dir__ of DocumentConverter object>':
                            print(f"{attr}: {value}")
                    except:
                        pass
            
            print("=== End Model Information ===\n")
            
            self.logger.info("Docling DocumentConverter initialized successfully")
            return converter
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Docling converter: {e}")
            raise
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract content from PDF using Docling's advanced understanding.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Comprehensive extraction results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.logger.error(f"PDF file not found: {pdf_path}")
            return None
        
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting Docling extraction from: {pdf_path.name}")
        self.logger.info(f"File size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
        start_time = datetime.now()
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output directories for this PDF
        pdf_output_dir = self.output_dir / pdf_path.stem
        self._create_output_directories(pdf_output_dir)
        
        extraction_results = {
            'pdf_name': pdf_path.name,
            'docling_version': 'latest',
            'extraction_timestamp': start_time.isoformat(),
            'document_analysis': {},
            'content_structure': {},
            'export_files': {},
            'comparison_metrics': {},
            'processing_success': True
        }
        
        try:
            # Convert document using Docling
            self.logger.info("Step 1/6: Converting document with Docling...")
            # Use convert_single for single documents (v1.20.0 API)
            result = self.converter.convert_single(str(pdf_path))
            
            # Extract DoclingDocument - result is the document itself in v1.20.0
            docling_doc = result
            self.logger.info(f"Document converted successfully. Total pages: {len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 'Unknown'}")
            
            # Analyze document structure
            self.logger.info("Step 2/6: Analyzing document structure...")
            extraction_results['document_analysis'] = self._analyze_document_structure(docling_doc)
            
            # Extract content with reading order
            self.logger.info("Step 3/6: Extracting content structure with reading order...")
            extraction_results['content_structure'] = self._extract_content_structure(
                docling_doc, pdf_output_dir
            )
            
            # Extract tables with advanced understanding
            self.logger.info("Step 4/6: Extracting advanced tables...")
            tables_info = self._extract_advanced_tables(docling_doc, pdf_output_dir)
            extraction_results['content_structure']['tables'] = tables_info
            
            # Extract formulas and mathematical content
            self.logger.info("Step 5/6: Extracting formulas and mathematical content...")
            formulas_info = self._extract_formulas(docling_doc, pdf_output_dir)
            extraction_results['content_structure']['formulas'] = formulas_info
            
            # Extract figures and images
            self.logger.info("Step 6/6: Extracting figures and images...")
            figures_info = self._extract_figures(docling_doc, pdf_output_dir)
            extraction_results['content_structure']['figures'] = figures_info
            
            # Export to different formats
            self.logger.info("Exporting to multiple formats (Markdown, JSON)...")
            export_files = self._export_to_formats(docling_doc, pdf_output_dir)
            extraction_results['export_files'] = export_files
            
            # Generate comparison metrics
            self.logger.info("Generating comparison metrics...")
            extraction_results['comparison_metrics'] = self._generate_comparison_metrics(docling_doc)
            
            # Update statistics
            self.stats['total_documents'] += 1
            self.stats['total_pages'] = len(docling_doc.pages) if hasattr(docling_doc, 'pages') else 0
            self.logger.info(f"Statistics updated: {self.stats['total_documents']} document(s) processed")
            
        except Exception as e:
            self.logger.error(f"Error during Docling extraction: {e}")
            extraction_results['processing_success'] = False
            extraction_results['error_message'] = str(e)
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        self.stats['processing_time'] = processing_time
        extraction_results['processing_time_seconds'] = processing_time
        
        # Save comprehensive results
        self._save_extraction_results(extraction_results, pdf_output_dir)
        
        self.logger.info(f"Docling extraction completed successfully")
        self.logger.info(f"Total processing time: {processing_time:.2f} seconds")
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log_statistics()
        self.logger.info(f"{'='*60}\n")
        
        return extraction_results
    
    def _create_output_directories(self, base_dir: Path):
        """Create organized output directory structure for Docling results."""
        directories = [
            'text', 'tables', 'formulas', 'figures', 'structure',
            'markdown', 'json', 'comparison', 'reading_order'
        ]
        for dir_name in directories:
            (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def _analyze_document_structure(self, docling_doc) -> Dict[str, Any]:
        """Analyze the overall document structure using Docling."""
        analysis = {
            'document_type': 'pdf',
            'total_pages': 0,
            'content_elements': [],
            'document_hierarchy': [],
            'metadata': {}
        }
        
        try:
            # Extract document metadata
            if hasattr(docling_doc, 'meta'):
                analysis['metadata'] = {
                    'title': getattr(docling_doc.meta, 'title', 'Unknown'),
                    'author': getattr(docling_doc.meta, 'author', 'Unknown'),
                    'creation_date': getattr(docling_doc.meta, 'creation_date', None),
                    'modification_date': getattr(docling_doc.meta, 'modification_date', None)
                }
                self.logger.debug(f"Metadata extracted: Title={analysis['metadata']['title']}, Author={analysis['metadata']['author']}")
            
            # Analyze page structure
            if hasattr(docling_doc, 'pages'):
                analysis['total_pages'] = len(docling_doc.pages)
                self.logger.info(f"  Total pages to analyze: {analysis['total_pages']}")
                
                for page_idx, page in enumerate(docling_doc.pages, 1):
                    page_info = {
                        'page_number': page_idx,
                        'elements': [],
                        'dimensions': getattr(page, 'dimensions', {})
                    }
                    
                    # Analyze page elements
                    if hasattr(page, 'elements'):
                        element_count = len(page.elements)
                        for element in page.elements:
                            element_info = {
                                'type': element.label if hasattr(element, 'label') else 'unknown',
                                'bbox': element.bbox.model_dump() if hasattr(element, 'bbox') else None,
                                'confidence': getattr(element, 'confidence', None)
                            }
                            page_info['elements'].append(element_info)
                        
                        self.logger.debug(f"  Page {page_idx}: {element_count} elements found")
                    
                    analysis['content_elements'].append(page_info)
            
            self.logger.info(f"Document structure analysis completed: {analysis['total_pages']} pages analyzed")
            
        except Exception as e:
            self.logger.error(f"Error analyzing document structure: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _extract_content_structure(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract content with proper reading order using Docling."""
        structure = {
            'reading_order': [],
            'content_blocks': [],
            'hierarchical_structure': [],
            'text_elements': []
        }
        
        try:
            # Extract content in reading order
            if hasattr(docling_doc, 'iterate_items'):
                reading_order_elements = []
                self.logger.info(f"  Extracting content with reading order...")
                
                for item in docling_doc.iterate_items():
                    element_data = {
                        'type': item.label if hasattr(item, 'label') else 'text',
                        'content': item.text if hasattr(item, 'text') else str(item),
                        'bbox': item.bbox.model_dump() if hasattr(item, 'bbox') else None,
                        'page': getattr(item, 'page', None),
                        'reading_order': len(reading_order_elements)
                    }
                    
                    reading_order_elements.append(element_data)
                    
                    # Save individual elements
                    if element_data['content'].strip():
                        element_file = output_dir / 'reading_order' / f"element_{len(reading_order_elements):04d}_{element_data['type']}.txt"
                        with open(element_file, 'w', encoding='utf-8') as f:
                            f.write(element_data['content'])
                    
                    # Log page-wise progress
                    if element_data['page'] is not None:
                        self.logger.debug(f"    Element {len(reading_order_elements)}: {element_data['type']} on page {element_data['page']}")
                
                structure['reading_order'] = reading_order_elements
                self.stats['reading_order_elements'] = len(reading_order_elements)
                self.logger.info(f"  Content structure extracted: {len(reading_order_elements)} elements in reading order")
            
            # Extract text content organized by structure
            self.logger.info(f"  Extracting structured text content...")
            full_text_content = self._extract_structured_text(docling_doc)
            
            # Save structured text
            text_file = output_dir / 'text' / 'structured_content.txt'
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(full_text_content)
            
            structure['full_text_file'] = str(text_file)
            self.logger.info(f"  Text content saved: {len(full_text_content)} characters")
            
        except Exception as e:
            self.logger.error(f"Error extracting content structure: {e}")
            structure['error'] = str(e)
        
        return structure
    
    def _extract_structured_text(self, docling_doc) -> str:
        """Extract text content with proper structure preservation using Docling v1.20.0 API."""
        try:
            # Use Docling v1.20.0 API methods
            if hasattr(docling_doc, 'render_as_markdown'):
                return docling_doc.render_as_markdown()
            elif hasattr(docling_doc, 'export_to_text'):
                return docling_doc.export_to_text()
            elif hasattr(docling_doc, 'text'):
                return docling_doc.text
            else:
                # Fallback: concatenate all text elements
                text_parts = []
                if hasattr(docling_doc, 'iterate_items'):
                    for item in docling_doc.iterate_items():
                        if hasattr(item, 'text') and item.text:
                            text_parts.append(item.text)
                return '\n\n'.join(text_parts)
        except Exception as e:
            self.logger.error(f"Error extracting structured text: {e}")
            return ""
    
    def _extract_advanced_tables(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract tables using Docling's advanced table understanding."""
        tables_info = {
            'count': 0,
            'tables': [],
            'extraction_method': 'docling_advanced'
        }
        
        try:
            tables = []
            self.logger.info(f"  Starting table extraction...")
            
            # Method 1: Extract tables using Docling's direct table detection
            if hasattr(docling_doc, 'tables') and docling_doc.tables:
                self.logger.info(f"  Found {len(docling_doc.tables)} table(s) using direct detection")
                for table_idx, table in enumerate(docling_doc.tables):
                    self.logger.debug(f"    Processing table {table_idx + 1}/{len(docling_doc.tables)}")
                    table_data = {
                        'table_id': f"docling_table_{table_idx:03d}",
                        'page': getattr(table, 'page', None),
                        'bbox': table.bbox.model_dump() if hasattr(table, 'bbox') else None,
                        'structure': None,
                        'content': None,
                        'csv_file': None
                    }
                    
                    # Extract table structure and content
                    if hasattr(table, 'to_dataframe'):
                        df = table.to_dataframe()
                        table_data['content'] = df.to_dict('records')
                        table_data['structure'] = {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'headers': list(df.columns)
                        }
                        
                        # Save as CSV
                        csv_file = output_dir / 'tables' / f"{table_data['table_id']}.csv"
                        df.to_csv(csv_file, index=False, encoding='utf-8')
                        table_data['csv_file'] = str(csv_file)
                        self.logger.debug(f"      Table {table_idx + 1}: {len(df)} rows × {len(df.columns)} columns - Saved to CSV")
                    
                    elif hasattr(table, 'data'):
                        # Handle raw table data
                        raw_data = table.data
                        if raw_data:
                            try:
                                df = pd.DataFrame(raw_data)
                                table_data['content'] = df.to_dict('records')
                                table_data['structure'] = {
                                    'rows': len(df),
                                    'columns': len(df.columns),
                                    'headers': list(df.columns)
                                }
                                
                                # Save as CSV
                                csv_file = output_dir / 'tables' / f"{table_data['table_id']}.csv"
                                df.to_csv(csv_file, index=False, encoding='utf-8')
                                table_data['csv_file'] = str(csv_file)
                                self.logger.debug(f"      Table {table_idx + 1}: {len(df)} rows × {len(df.columns)} columns - Saved to CSV")
                            except Exception as e:
                                self.logger.warning(f"      Table {table_idx + 1}: Failed to process table data: {e}")
                                table_data['content'] = raw_data
                                table_data['error'] = str(e)
                    
                    elif hasattr(table, 'text') or hasattr(table, 'content'):
                        # Handle text-based table content
                        table_text = getattr(table, 'text', None) or getattr(table, 'content', str(table))
                        if table_text:
                            # Save as text file
                            text_file = output_dir / 'tables' / f"{table_data['table_id']}.txt"
                            with open(text_file, 'w', encoding='utf-8') as f:
                                f.write(table_text)
                            table_data['text_file'] = str(text_file)
                            table_data['content'] = table_text
                            self.logger.debug(f"      Table {table_idx + 1}: Saved as text file")
                    
                    tables.append(table_data)
                    
                    # Save individual table metadata
                    table_json = output_dir / 'tables' / f"{table_data['table_id']}_metadata.json"
                    with open(table_json, 'w', encoding='utf-8') as f:
                        json.dump(table_data, f, indent=2, ensure_ascii=False, default=str)
            
            # Method 2: Extract tables from document structure if no direct tables found
            if not tables and hasattr(docling_doc, 'iterate_items'):
                self.logger.info(f"  No direct tables found, searching in document structure...")
                table_idx = 0
                for item in docling_doc.iterate_items():
                    # Look for table-like structures
                    if (hasattr(item, 'label') and item.label and 
                        ('table' in item.label.lower() or 'grid' in item.label.lower())):
                        
                        self.logger.debug(f"    Found table-like structure: {item.label}")
                        table_data = {
                            'table_id': f"doc_table_{table_idx:03d}",
                            'page': getattr(item, 'page', None),
                            'bbox': item.bbox.model_dump() if hasattr(item, 'bbox') else None,
                            'structure': None,
                            'content': getattr(item, 'text', str(item)),
                            'source': 'document_structure'
                        }
                        
                        # Save table as text file
                        text_file = output_dir / 'tables' / f"{table_data['table_id']}.txt"
                        with open(text_file, 'w', encoding='utf-8') as f:
                            f.write(table_data['content'])
                        table_data['text_file'] = str(text_file)
                        
                        # Save table metadata
                        table_json = output_dir / 'tables' / f"{table_data['table_id']}_metadata.json"
                        with open(table_json, 'w', encoding='utf-8') as f:
                            json.dump(table_data, f, indent=2, ensure_ascii=False, default=str)
                        
                        tables.append(table_data)
                        table_idx += 1
            
            # Method 3: Look for table-like patterns in markdown text
            if not tables:
                self.logger.info(f"  No tables found in document structure, extracting from markdown patterns...")
                markdown_text = self._extract_structured_text(docling_doc)
                table_patterns = self._extract_tables_from_markdown(markdown_text, output_dir)
                tables.extend(table_patterns)
            
            tables_info['count'] = len(tables)
            tables_info['tables'] = tables
            self.stats['tables_extracted'] = len(tables)
            
            self.logger.info(f"  Table extraction completed: {len(tables)} table(s) found")
            
        except Exception as e:
            self.logger.error(f"Error extracting advanced tables: {e}")
            tables_info['error'] = str(e)
        
        return tables_info
    
    def _extract_tables_from_markdown(self, markdown_text: str, output_dir: Path) -> List[Dict[str, Any]]:
        """Extract tables from markdown text using table patterns."""
        tables = []
        if not markdown_text:
            return tables
        
        lines = markdown_text.split('\n')
        table_idx = 0
        current_table = []
        in_table = False
        
        for line in lines:
            line = line.strip()
            # Check if line looks like a table row (contains |)
            if '|' in line and line.count('|') >= 2:
                current_table.append(line)
                in_table = True
            elif in_table and current_table:
                # End of table, process it
                if len(current_table) >= 2:  # At least header and one row
                    table_data = {
                        'table_id': f"markdown_table_{table_idx:03d}",
                        'page': None,
                        'bbox': None,
                        'content': current_table,
                        'source': 'markdown_pattern'
                    }
                    
                    # Save table as text file
                    text_file = output_dir / 'tables' / f"{table_data['table_id']}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(current_table))
                    table_data['text_file'] = str(text_file)
                    
                    # Try to parse as CSV
                    try:
                        # Convert markdown table to CSV format
                        csv_lines = []
                        for table_line in current_table:
                            if '|' in table_line:
                                # Remove leading/trailing |, split by |, and clean
                                cells = [cell.strip() for cell in table_line.split('|')[1:-1]]
                                if cells and not all(cell.replace('-', '').strip() == '' for cell in cells):
                                    csv_lines.append(','.join(f'"{cell}"' for cell in cells))
                        
                        if csv_lines:
                            csv_file = output_dir / 'tables' / f"{table_data['table_id']}.csv"
                            with open(csv_file, 'w', encoding='utf-8') as f:
                                f.write('\n'.join(csv_lines))
                            table_data['csv_file'] = str(csv_file)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert markdown table to CSV: {e}")
                    
                    # Save metadata
                    table_json = output_dir / 'tables' / f"{table_data['table_id']}_metadata.json"
                    with open(table_json, 'w', encoding='utf-8') as f:
                        json.dump(table_data, f, indent=2, ensure_ascii=False, default=str)
                    
                    tables.append(table_data)
                    table_idx += 1
                
                current_table = []
                in_table = False
        
        # Handle last table if file ends while in table
        if in_table and current_table and len(current_table) >= 2:
            table_data = {
                'table_id': f"markdown_table_{table_idx:03d}",
                'page': None,
                'bbox': None,
                'content': current_table,
                'source': 'markdown_pattern'
            }
            
            text_file = output_dir / 'tables' / f"{table_data['table_id']}.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(current_table))
            table_data['text_file'] = str(text_file)
            
            tables.append(table_data)
        
        return tables
    
    def _extract_formulas(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract mathematical formulas and equations."""
        formulas_info = {
            'count': 0,
            'formulas': [],
            'extraction_method': 'docling_formula_detection'
        }
        
        try:
            formulas = []
            self.logger.info(f"  Starting formula extraction...")
            
            # Method 1: Look for direct formula elements
            if hasattr(docling_doc, 'iterate_items'):
                formula_idx = 0
                for item in docling_doc.iterate_items():
                    # Check if item is a formula or contains mathematical notation
                    if (hasattr(item, 'label') and item.label and
                        any(math_term in item.label.lower() for math_term in ['formula', 'equation', 'math']) or
                        (hasattr(item, 'text') and item.text and self._contains_math_notation(item.text))):
                        
                        self.logger.debug(f"    Found formula: {item.label if hasattr(item, 'label') else 'math notation'} on page {getattr(item, 'page', '?')}")
                        formula_data = {
                            'formula_id': f"formula_{formula_idx:03d}",
                            'content': item.text if hasattr(item, 'text') else str(item),
                            'bbox': item.bbox.model_dump() if hasattr(item, 'bbox') else None,
                            'page': getattr(item, 'page', None),
                            'type': getattr(item, 'label', 'mathematical_expression')
                        }
                        
                        # Save formula content
                        formula_file = output_dir / 'formulas' / f"{formula_data['formula_id']}.txt"
                        with open(formula_file, 'w', encoding='utf-8') as f:
                            f.write(f"Formula ID: {formula_data['formula_id']}\n")
                            f.write(f"Type: {formula_data['type']}\n")
                            f.write(f"Page: {formula_data['page']}\n")
                            f.write(f"Content: {formula_data['content']}\n")
                        formula_data['file'] = str(formula_file)
                        
                        formulas.append(formula_data)
                        formula_idx += 1
            
            # Method 2: Scan text content for mathematical expressions
            if not formulas:
                self.logger.debug(f"    No direct formulas found, scanning text for mathematical notation...")
                markdown_text = self._extract_structured_text(docling_doc)
                if markdown_text:
                    lines = markdown_text.split('\n')
                    formula_idx = 0
                    for line_idx, line in enumerate(lines):
                        if self._contains_math_notation(line):
                            self.logger.debug(f"    Found math notation on line {line_idx + 1}")
                            formula_data = {
                                'formula_id': f"text_formula_{formula_idx:03d}",
                                'content': line.strip(),
                                'line_number': line_idx + 1,
                                'type': 'text_mathematical_expression',
                                'source': 'text_scan'
                            }
                            
                            # Save formula content
                            formula_file = output_dir / 'formulas' / f"{formula_data['formula_id']}.txt"
                            with open(formula_file, 'w', encoding='utf-8') as f:
                                f.write(f"Formula ID: {formula_data['formula_id']}\n")
                                f.write(f"Line: {formula_data['line_number']}\n")
                                f.write(f"Content: {formula_data['content']}\n")
                            formula_data['file'] = str(formula_file)
                            
                            formulas.append(formula_data)
                            formula_idx += 1
            
            formulas_info['count'] = len(formulas)
            formulas_info['formulas'] = formulas
            self.stats['formulas_detected'] = len(formulas)
            
            self.logger.info(f"  Formula extraction completed: {len(formulas)} formula(s) detected")
            
        except Exception as e:
            self.logger.error(f"Error extracting formulas: {e}")
            formulas_info['error'] = str(e)
        
        return formulas_info
    
    def _contains_math_notation(self, text: str) -> bool:
        """Check if text contains mathematical notation."""
        if not text:
            return False
        
        math_indicators = [
            '∑', '∫', '∂', '∆', '∇', '±', '≤', '≥', '≠', '≈', '∞',
            '∝', '∴', '∵', '∈', '∉', '⊂', '⊃', '∪', '∩', '∅',
            '√', '∛', '∜', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹',
            '₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉',
            'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'φ', 'ψ', 'ω',
            '\\frac', '\\sum', '\\int', '\\partial', '\\delta', '\\nabla'
        ]
        
        return any(indicator in text for indicator in math_indicators)
    
    def _extract_figures(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract figures and images with metadata."""
        figures_info = {
            'count': 0,
            'figures': [],
            'extraction_method': 'docling_figure_detection'
        }
        
        try:
            figures = []
            self.logger.info(f"  Starting figure extraction...")
            
            # Method 1: Check for direct figure/picture attributes on document
            figure_attributes = ['pictures', 'figures', 'images', '_figures', '_pictures']
            
            for attr in figure_attributes:
                if hasattr(docling_doc, attr):
                    pictures = getattr(docling_doc, attr, [])
                    if pictures:
                        self.logger.info(f"  Found {len(pictures)} figure(s) using '{attr}' attribute")
                        for fig_idx, figure in enumerate(pictures):
                            self.logger.debug(f"    Processing figure {fig_idx + 1}/{len(pictures)}")
                            figure_data = {
                                'figure_id': f"figure_{fig_idx:03d}",
                                'bbox': figure.bbox.model_dump() if hasattr(figure, 'bbox') else None,
                                'page': getattr(figure, 'page', None),
                                'caption': getattr(figure, 'caption', None) or getattr(figure, 'text', None),
                                'image_file': None,
                                'source': attr
                            }
                            
                            # Save figure image if available
                            if hasattr(figure, 'image'):
                                try:
                                    image_file = output_dir / 'figures' / f"{figure_data['figure_id']}.png"
                                    # Save the image data
                                    with open(image_file, 'wb') as f:
                                        if hasattr(figure.image, 'data'):
                                            f.write(figure.image.data)
                                        elif isinstance(figure.image, bytes):
                                            f.write(figure.image)
                                    figure_data['image_file'] = str(image_file)
                                    self.logger.debug(f"      Figure {fig_idx + 1}: Image saved on page {figure_data['page']}")
                                except Exception as e:
                                    self.logger.warning(f"      Figure {fig_idx + 1}: Failed to save image - {e}")
                            
                            figures.append(figure_data)
                        if figures:
                            break  # Found figures, no need to check other attributes
            
            # Method 2: Extract figures from document structure by iterating through elements
            # This is the most reliable method for Docling v1.20.0
            if not figures and hasattr(docling_doc, 'iterate_items'):
                self.logger.debug(f"    Searching for figures in document structure via iterate_items()...")
                fig_idx = 0
                for item in docling_doc.iterate_items():
                    # Check for figure/image elements by label
                    item_label = getattr(item, 'label', None)
                    if item_label:
                        item_label_lower = str(item_label).lower()
                        is_figure = any(term in item_label_lower for term in ['figure', 'image', 'picture', 'photo', 'diagram', 'illustration', 'graphic'])
                        
                        if is_figure:
                            self.logger.debug(f"    Found figure element: {item.label} on page {getattr(item, 'page', '?')}")
                            figure_data = {
                                'figure_id': f"figure_{fig_idx:03d}",
                                'bbox': item.bbox.model_dump() if hasattr(item, 'bbox') else None,
                                'page': getattr(item, 'page', None),
                                'caption': getattr(item, 'text', None),
                                'label': str(item.label),
                                'content': str(item),
                                'source': 'iterate_items'
                            }
                            
                            # Try to extract image if available
                            if hasattr(item, 'image'):
                                try:
                                    image_file = output_dir / 'figures' / f"{figure_data['figure_id']}.png"
                                    with open(image_file, 'wb') as f:
                                        if hasattr(item.image, 'data'):
                                            f.write(item.image.data)
                                        elif isinstance(item.image, bytes):
                                            f.write(item.image)
                                    figure_data['image_file'] = str(image_file)
                                    self.logger.debug(f"    Saved image for figure {fig_idx}")
                                except Exception as e:
                                    self.logger.warning(f"    Failed to save image for figure {fig_idx}: {e}")
                            
                            # Save figure metadata as text/json
                            metadata_file = output_dir / 'figures' / f"{figure_data['figure_id']}_metadata.json"
                            with open(metadata_file, 'w', encoding='utf-8') as f:
                                json.dump(figure_data, f, indent=2, ensure_ascii=False, default=str)
                            
                            figures.append(figure_data)
                            fig_idx += 1
            
            # Method 3: Look through document blocks for figure content
            if not figures and hasattr(docling_doc, 'blocks'):
                self.logger.debug(f"    Searching for figures in document blocks...")
                fig_idx = 0
                
                for block in docling_doc.blocks:
                    block_type = getattr(block, 'class_name', None) or type(block).__name__
                    self.logger.debug(f"    Examining block type: {block_type}")
                    
                    # Look for Picture/Figure blocks
                    if any(term in str(block_type).lower() for term in ['picture', 'figure', 'image']):
                        self.logger.debug(f"    Found figure block: {block_type}")
                        
                        figure_data = {
                            'figure_id': f"figure_{fig_idx:03d}",
                            'block_type': str(block_type),
                            'bbox': block.bbox.model_dump() if hasattr(block, 'bbox') else None,
                            'page': getattr(block, 'page', None),
                            'caption': getattr(block, 'caption', None),
                            'text': getattr(block, 'text', None),
                            'source': 'blocks'
                        }
                        
                        # Try to extract image data
                        if hasattr(block, 'image'):
                            try:
                                image_file = output_dir / 'figures' / f"{figure_data['figure_id']}.png"
                                with open(image_file, 'wb') as f:
                                    if hasattr(block.image, 'data'):
                                        f.write(block.image.data)
                                    elif isinstance(block.image, bytes):
                                        f.write(block.image)
                                figure_data['image_file'] = str(image_file)
                                self.logger.debug(f"    Saved image for figure {fig_idx} from block")
                            except Exception as e:
                                self.logger.warning(f"    Failed to extract image from block: {e}")
                        
                        # Save metadata
                        metadata_file = output_dir / 'figures' / f"{figure_data['figure_id']}_metadata.json"
                        with open(metadata_file, 'w', encoding='utf-8') as f:
                            json.dump(figure_data, f, indent=2, ensure_ascii=False, default=str)
                        
                        figures.append(figure_data)
                        fig_idx += 1
            
            # Method 4: Scan markdown for embedded images (fallback)
            if not figures:
                self.logger.debug(f"    Searching for figures in markdown content...")
                markdown_text = self._extract_structured_text(docling_doc)
                fig_idx = 0
                
                # Look for markdown image syntax: ![alt](url) or <img> tags
                import re
                
                # Find markdown images
                md_image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
                for match in re.finditer(md_image_pattern, markdown_text):
                    alt_text, image_url = match.groups()
                    self.logger.debug(f"    Found markdown image: {alt_text}")
                    
                    figure_data = {
                        'figure_id': f"figure_{fig_idx:03d}",
                        'alt_text': alt_text,
                        'url': image_url,
                        'source': 'markdown'
                    }
                    
                    figures.append(figure_data)
                    fig_idx += 1
                
                # Find HTML img tags
                html_image_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*alt=["\']([^"\']*)["\']'
                for match in re.finditer(html_image_pattern, markdown_text):
                    image_url, alt_text = match.groups()
                    self.logger.debug(f"    Found HTML image: {alt_text}")
                    
                    figure_data = {
                        'figure_id': f"figure_{fig_idx:03d}",
                        'alt_text': alt_text,
                        'url': image_url,
                        'source': 'html'
                    }
                    
                    figures.append(figure_data)
                    fig_idx += 1
            
            figures_info['count'] = len(figures)
            figures_info['figures'] = figures
            self.stats['figures_detected'] = len(figures)
            
            self.logger.info(f"  Figure extraction completed: {len(figures)} figure(s) detected")
            
            if len(figures) == 0:
                self.logger.warning(f"  No figures detected. Document may not have images or they may not be extractable.")
                self.logger.debug(f"  Available document attributes: {[attr for attr in dir(docling_doc) if not attr.startswith('_')][:10]}...")
            
        except Exception as e:
            self.logger.error(f"Error extracting figures: {e}", exc_info=True)
            figures_info['error'] = str(e)
        
        return figures_info
    
    def _export_to_formats(self, docling_doc, output_dir: Path) -> Dict[str, str]:
        """Export document to various formats (Markdown, JSON, etc.)."""
        export_files = {}
        
        try:
            self.logger.info(f"  Starting format exports...")
            
            # Export to Markdown using v1.20.0 API
            if hasattr(docling_doc, 'render_as_markdown'):
                self.logger.debug(f"    Exporting to Markdown format...")
                markdown_content = docling_doc.render_as_markdown()
                markdown_file = output_dir / 'markdown' / 'document.md'
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                export_files['markdown'] = str(markdown_file)
                self.stats['export_formats'].append('markdown')
                self.logger.debug(f"    Markdown export complete ({len(markdown_content)} chars)")
            elif hasattr(docling_doc, 'export_to_markdown'):
                self.logger.debug(f"    Exporting to Markdown format...")
                markdown_content = docling_doc.export_to_markdown()
                markdown_file = output_dir / 'markdown' / 'document.md'
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                export_files['markdown'] = str(markdown_file)
                self.stats['export_formats'].append('markdown')
                self.logger.debug(f"    Markdown export complete ({len(markdown_content)} chars)")
            
            # Export to JSON/DocTags using v1.20.0 API
            if hasattr(docling_doc, 'render_as_doctags'):
                self.logger.debug(f"    Exporting to DocTags format...")
                doctags_content = docling_doc.render_as_doctags()
                doctags_file = output_dir / 'json' / 'document_doctags.xml'
                with open(doctags_file, 'w', encoding='utf-8') as f:
                    f.write(doctags_content)
                export_files['doctags'] = str(doctags_file)
                self.stats['export_formats'].append('doctags')
                self.logger.debug(f"    DocTags export complete")
            
            # Export to JSON (fallback)
            if hasattr(docling_doc, 'export_to_dict'):
                self.logger.debug(f"    Exporting to JSON format...")
                json_content = docling_doc.export_to_dict()
                json_file = output_dir / 'json' / 'document.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False, default=str)
                export_files['json'] = str(json_file)
                self.stats['export_formats'].append('json')
                self.logger.debug(f"    JSON export complete")
            
            # Export structured text
            self.logger.debug(f"    Exporting full document as text...")
            text_content = self._extract_structured_text(docling_doc)
            text_file = output_dir / 'text' / 'full_document.txt'
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            export_files['text'] = str(text_file)
            self.logger.debug(f"    Text export complete ({len(text_content)} chars)")
            
            self.logger.info(f"  Format exports complete: {list(export_files.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to formats: {e}")
            export_files['error'] = str(e)
        
        return export_files
    
    def _generate_comparison_metrics(self, docling_doc) -> Dict[str, Any]:
        """Generate metrics for comparison with traditional methods."""
        metrics = {
            'docling_advantages': [],
            'content_analysis': {},
            'structure_quality': {},
            'extraction_completeness': {}
        }
        
        try:
            # Analyze content quality
            metrics['content_analysis'] = {
                'reading_order_preserved': self.stats['reading_order_elements'] > 0,
                'tables_with_structure': self.stats['tables_extracted'],
                'formulas_detected': self.stats['formulas_detected'],
                'figures_extracted': self.stats['figures_detected']
            }
            
            # Document structure quality
            metrics['structure_quality'] = {
                'hierarchical_extraction': True,  # Docling preserves hierarchy
                'semantic_understanding': True,   # Advanced semantic analysis
                'layout_preservation': True,      # Layout information retained
                'cross_page_continuity': True     # Handles content across pages
            }
            
            # Docling advantages over traditional methods
            advantages = [
                "Preserves reading order and document hierarchy",
                "Advanced table structure recognition",
                "Mathematical formula detection",
                "Unified document model (DoclingDocument)",
                "Multiple export formats (Markdown, JSON)",
                "Semantic understanding of document elements",
                "Better handling of complex layouts"
            ]
            metrics['docling_advantages'] = advantages
            
        except Exception as e:
            self.logger.error(f"Error generating comparison metrics: {e}")
            metrics['error'] = str(e)
        
        return metrics
    
    def _save_extraction_results(self, results: Dict[str, Any], output_dir: Path):
        """Save comprehensive Docling extraction results."""
        # Save main results
        results_file = output_dir / 'docling_extraction_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save comparison analysis
        comparison_file = output_dir / 'comparison' / 'docling_vs_traditional.json'
        comparison_data = {
            'docling_metrics': results.get('comparison_metrics', {}),
            'extraction_stats': self.stats,
            'advantages_summary': results.get('comparison_metrics', {}).get('docling_advantages', []),
            'processing_time': results.get('processing_time_seconds', 0)
        }
        
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Docling extraction results saved to {results_file}")
        self.logger.info(f"Comparison analysis saved to {comparison_file}")
    
    def _log_statistics(self):
        """Log comprehensive Docling extraction statistics."""
        self.logger.info("="*60)
        self.logger.info("EXTRACTION STATISTICS SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Documents processed:      {self.stats['total_documents']}")
        self.logger.info(f"Total pages analyzed:     {self.stats['total_pages']}")
        self.logger.info(f"Reading order elements:   {self.stats['reading_order_elements']}")
        self.logger.info(f"Tables extracted:         {self.stats['tables_extracted']}")
        self.logger.info(f"Formulas detected:        {self.stats['formulas_detected']}")
        self.logger.info(f"Figures detected:         {self.stats['figures_detected']}")
        self.logger.info(f"Export formats used:      {', '.join(self.stats['export_formats']) if self.stats['export_formats'] else 'None'}")
        self.logger.info(f"Total processing time:    {self.stats['processing_time']:.2f} seconds")
        self.logger.info("="*60)


def main():
    """Main function to demonstrate Docling-based extraction with path and PDF name arguments."""
    import argparse
    import sys
    from pathlib import Path
    
    # Add project root to path to import from pilots
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
    from update_provenance import update_parser_provenance
    print("=== Docling Advanced PDF Understanding ===")
    print("Usage: python src/parse/docling_extractor.py <path> --pdf <pdf_name>")
    print("Example: python src/parse/docling_extractor.py FINTBX/pdf/split_pdfs --pdf fintbx_part_001.pdf")
    print()
    print("Output structure: data/parsed/{Company}/[pdf_name]/")
    print("  ├── text/           - Structured text content")
    print("  ├── tables/         - Advanced table extraction")
    print("  ├── formulas/       - Mathematical formulas")
    print("  ├── figures/        - Extracted figures")
    print("  ├── structure/      - Document structure analysis")
    print("  ├── markdown/       - Markdown export")
    print("  ├── json/           - JSON export")
    print("  ├── comparison/     - Comparison with traditional methods")
    print("  └── reading_order/  - Reading order elements")
    print()

    parser = argparse.ArgumentParser(description="Docling PDF Extraction from split PDFs")
    parser.add_argument("path", type=str, help="Relative path to PDF directory (e.g., FINTBX/pdf/split_pdfs)")
    parser.add_argument("--pdf", type=str, required=True, help="PDF file name to process (e.g., fintbx_part_001.pdf)")
    args = parser.parse_args()

    # Extract company ticker from the path
    # Path format: FINTBX/pdf/split_pdfs or similar
    path_parts = args.path.split('/')
    company_ticker = path_parts[0] if path_parts else "Unknown"
    
    # Build full PDF path: data/raw/{path}/{pdf_name}
    pdf_file = Path(f"data/raw/{args.path}/{args.pdf}")
    
    # Verify the PDF file exists
    if not pdf_file.exists():
        print(f"✗ PDF file not found: {pdf_file}")
        print(f"Expected location: {pdf_file}")
        return

    try:
        # Initialize extractor with company ticker
        extractor = DoclingExtractor(output_dir=f"data/parsed/{company_ticker}", company_ticker=company_ticker)

        print(f"\nProcessing PDF from path: {args.path}")
        print(f"PDF file: {args.pdf}")
        print(f"Company ticker: {company_ticker}")
        print(f"Full path: {pdf_file}")
        print()

        # Process the specific PDF file
        print(f"Processing with Docling: {pdf_file.name}")
        results = extractor.extract_from_pdf(pdf_file)

        if results and results.get('processing_success', False):
            print(f"\n✓ Docling extraction completed for {pdf_file.name}")
            print(f"  Pages processed: {results['document_analysis'].get('total_pages', 0)}")
            print(f"  Reading order elements: {len(results['content_structure'].get('reading_order', []))}")
            print(f"  Tables extracted: {results['content_structure']['tables']['count']}")
            print(f"  Formulas detected: {results['content_structure']['formulas']['count']}")
            print(f"  Export formats: {list(results['export_files'].keys())}")
            print(f"  Processing time: {results['processing_time_seconds']:.2f}s")

            # Automatically collect all output file paths
            import os
            output_dir = f"data/parsed/{company_ticker}/{pdf_file.stem}"
            parsed_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), start=".")
                    parsed_files.append(rel_path.replace("\\", "/"))
            parsed_files_summary = f"All output files generated for {pdf_file.name} including extraction results, text, markdown, doctags, comparison, and all table files."

            # Update parser provenance after successful extraction
            parser_info = {
                "pdf_path": str(pdf_file),
                "pdf_name": results.get("pdf_name", ""),
                "parsing_time": results.get("extraction_timestamp", ""),
                "pages_processed": results['document_analysis'].get('total_pages', 0),
                "reading_order_elements": len(results['content_structure'].get('reading_order', [])),
                "tables_extracted": results['content_structure']['tables']['count'],
                "formulas_detected": results['content_structure']['formulas']['count'],
                "export_formats": list(results['export_files'].keys()),
                "processing_time_seconds": results['processing_time_seconds']
            }
            update_parser_provenance(
                company_ticker,
                [parser_info],
                parsed_files=parsed_files,
                parsed_files_summary=parsed_files_summary
            )
        else:
            print(f"\n✗ Docling extraction failed for {pdf_file.name}")
            if results and 'error_message' in results:
                print(f"  Error: {results['error_message']}")

    except ImportError as e:
        print("Docling is not installed. To install Docling:")
        print("1. pip install docling")
        print("2. Note: Requires Python 3.9+ and may need system dependencies")
        print(f"Error: {e}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()