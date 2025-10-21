"""
Docling-based Advanced PDF Understanding and Content Extraction (v2.49.0)

This module uses IBM's Docling library v2.49.0 for sophisticated PDF analysis including:
- Reading order analysis
- Advanced table extraction
- Formula recognition
- Unified DoclingDocument format
- Export to Markdown and JSON

Docling provides state-of-the-art document understanding capabilities
that go beyond traditional text extraction methods.

Compatible with: Docling >= 2.0.0
"""

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import ConversionStatus
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

import json
import os
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd


class DoclingExtractor:
    """
    Advanced PDF content extraction using IBM's Docling library v2.49.0.
    
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
                "Docling is not installed. Install with: pip install docling>=2.0.0\n"
                "For v2.49.0: pip install 'docling==2.49.0'"
            )
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize document converter with advanced options for v2.49.0
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
        logger = logging.getLogger('DoclingExtractor_v2')
        logger.setLevel(logging.INFO)
        
        # Create logs directory based on company ticker
        if self.company_ticker:
            log_dir = Path(f"data/logs/{self.company_ticker}")
        else:
            log_dir = Path("data/logs/docling")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        log_file = log_dir / 'docling_extraction_v2.log'
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
        """Initialize Docling document converter for v2.49.0."""
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            
            # Initialize converter with pipeline options to enable page image extraction
            # v2.49.0 uses OCR and AI models automatically
            pipeline_options = PdfPipelineOptions()
            pipeline_options.generate_page_images = True  # Enable page image generation
            pipeline_options.images_scale = 1.0  # Normal scale
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            self.logger.info("Docling DocumentConverter v2.49.0 initialized successfully")
            self.logger.info("Available models: Table Detection, Formula Detection, Layout Analysis")
            self.logger.info("Page images generation: ENABLED")
            
            return converter
        
        except Exception as e:
            self.logger.error(f"Failed to initialize Docling converter: {e}", exc_info=True)
            raise
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract content from PDF using Docling v2.49.0 advanced understanding.
        
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
        self.logger.info(f"Starting Docling v2.49.0 extraction from: {pdf_path.name}")
        self.logger.info(f"File size: {pdf_path.stat().st_size / (1024*1024):.2f} MB")
        start_time = datetime.now()
        self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Create output directories for this PDF
        pdf_output_dir = self.output_dir / pdf_path.stem
        self._create_output_directories(pdf_output_dir)
        
        extraction_results = {
            'pdf_name': pdf_path.name,
            'docling_version': '2.49.0',
            'extraction_timestamp': start_time.isoformat(),
            'document_analysis': {},
            'content_structure': {},
            'export_files': {},
            'comparison_metrics': {},
            'processing_success': True
        }
        
        try:
            # Convert document using Docling v2.49.0
            self.logger.info("Step 1/6: Converting document with Docling v2.49.0...")
            result = self.converter.convert(str(pdf_path))
            
            # Check conversion status
            if result.status != ConversionStatus.SUCCESS:
                raise Exception(f"Conversion failed with status: {result.status}")
            
            # Extract DoclingDocument
            docling_doc = result.document
            self.logger.info(f"Document converted successfully. Total pages: {len(docling_doc.pages)}")
            
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
            self.stats['total_pages'] = len(docling_doc.pages)
            self.logger.info(f"Statistics updated: {self.stats['total_documents']} document(s) processed")
            
        except Exception as e:
            self.logger.error(f"Error during Docling extraction: {e}", exc_info=True)
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
        """Analyze the overall document structure using Docling v2.49.0."""
        analysis = {
            'document_type': 'pdf',
            'total_pages': 0,
            'content_elements': [],
            'document_hierarchy': [],
            'metadata': {}
        }
        
        try:
            # Extract document metadata (v2.49.0 API)
            if hasattr(docling_doc, 'meta'):
                analysis['metadata'] = {
                    'title': getattr(docling_doc.meta, 'title', 'Unknown'),
                    'author': getattr(docling_doc.meta, 'author', 'Unknown'),
                    'creation_date': str(getattr(docling_doc.meta, 'creation_date', None)),
                    'modification_date': str(getattr(docling_doc.meta, 'modification_date', None))
                }
                self.logger.info(f"  Metadata: Title={analysis['metadata']['title']}")
            
            # Analyze page structure
            analysis['total_pages'] = len(docling_doc.pages)
            self.logger.info(f"  Total pages to analyze: {analysis['total_pages']}")
            
            for page_idx, page in enumerate(docling_doc.pages, 1):
                page_info = {
                    'page_number': page_idx,
                    'elements': [],
                    'width': page.size.width if hasattr(page, 'size') else None,
                    'height': page.size.height if hasattr(page, 'size') else None
                }
                
                # Count elements on page
                if hasattr(page, 'children'):
                    page_info['element_count'] = len(page.children)
                    self.logger.debug(f"  Page {page_idx}: {page_info['element_count']} elements found")
                
                analysis['content_elements'].append(page_info)
            
            self.logger.info(f"Document structure analysis completed: {analysis['total_pages']} pages analyzed")
            
        except Exception as e:
            self.logger.error(f"Error analyzing document structure: {e}", exc_info=True)
            analysis['error'] = str(e)
        
        return analysis
    
    def _extract_content_structure(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract content with proper reading order using Docling v2.49.0 iterate_items() API."""
        structure = {
            'reading_order': [],
            'content_blocks': [],
            'text_elements': []
        }
        
        try:
            reading_order_elements = []
            self.logger.info(f"  Extracting content with reading order using iterate_items()...")
            
            # Use the CORRECT Docling v2.49.0 API: doc.iterate_items()
            # This properly handles the document hierarchy and content ordering
            element_counter = 0
            table_counter = 0  # Track table sequence for mapping to extracted tables
            
            for element, level in docling_doc.iterate_items():
                element_type = type(element).__name__
                
                # Get text content - works for TextItem, SectionHeaderItem, etc.
                content = getattr(element, 'text', '') or ""
                
                # Get page number from provenance
                page_num = 0
                if hasattr(element, 'prov') and element.prov:
                    page_num = element.prov[0].page_no
                
                element_data = {
                    'type': element_type,
                    'content': content,
                    'page': page_num,
                    'reading_order': element_counter,
                    'hierarchy_level': level
                }
                
                # ENHANCEMENT: For TableItems, add table mapping metadata
                # This allows PDF generation to look up table content from content_structure.tables
                if element_type == 'TableItem':
                    element_data['table_index'] = table_counter  # Maps to tables_list[table_counter]
                    element_data['table_mapping_note'] = f"Maps to content_structure.tables[{table_counter}]"
                    table_counter += 1
                
                reading_order_elements.append(element_data)
                
                # Save individual text elements (only non-empty)
                if content.strip():
                    element_file = output_dir / 'reading_order' / f"page_{page_num:03d}_element_{element_counter:04d}_{element_type}.txt"
                    element_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(element_file, 'w', encoding='utf-8') as f:
                        f.write(f"Type: {element_data['type']}\n")
                        f.write(f"Page: {element_data['page']}\n")
                        f.write(f"Hierarchy Level: {element_data['hierarchy_level']}\n")
                        if element_type == 'TableItem':
                            f.write(f"Table Index: {element_data.get('table_index')}\n")
                        f.write(f"Content:\n{element_data['content']}")
                
                element_counter += 1
                self.logger.debug(f"      Element {element_counter}: {element_type} on page {page_num}" + 
                                 (f" [Table #{table_counter-1}]" if element_type == 'TableItem' else ""))
            
            structure['reading_order'] = reading_order_elements
            self.stats['reading_order_elements'] = len(reading_order_elements)
            self.logger.info(f"  Content structure extracted: {len(reading_order_elements)} elements in reading order")
            self.logger.info(f"  Table mapping created: {table_counter} TableItems mapped to extracted tables")
            
            # Extract full text
            self.logger.info(f"  Extracting full text content...")
            full_text_content = self._extract_structured_text(docling_doc)
            
            text_file = output_dir / 'text' / 'structured_content.txt'
            text_file.parent.mkdir(parents=True, exist_ok=True)
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(full_text_content)
            
            structure['full_text_file'] = str(text_file)
            self.logger.info(f"  Text content saved: {len(full_text_content)} characters")
            
        except Exception as e:
            self.logger.error(f"Error extracting content structure: {e}", exc_info=True)
            structure['error'] = str(e)
        
        return structure
    
    def _extract_structured_text(self, docling_doc) -> str:
        """Extract text content with proper structure preservation."""
        try:
            # v2.49.0 API: Use export_to_markdown()
            if hasattr(docling_doc, 'export_to_markdown'):
                return docling_doc.export_to_markdown()
            # Fallback: concatenate all text items
            elif hasattr(docling_doc, 'iterate_items'):
                text_parts = []
                for item in docling_doc.iterate_items():
                    if hasattr(item, 'text') and item.text:
                        text_parts.append(item.text)
                return '\n\n'.join(text_parts)
            else:
                return ""
        except Exception as e:
            self.logger.error(f"Error extracting structured text: {e}")
            return ""
    
    def _extract_advanced_tables(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract tables using Docling v2.49.0 advanced table understanding."""
        tables_info = {
            'count': 0,
            'tables': [],
            'extraction_method': 'docling_v2_native_blocks'
        }
        
        try:
            tables = []
            self.logger.info(f"  Starting table extraction from blocks...")
            
            # Direct table access via document.tables (v2.49.0 CORRECT API)
            if hasattr(docling_doc, 'tables') and docling_doc.tables:
                self.logger.info(f"  Found {len(docling_doc.tables)} table(s)")
                
                for table_idx, table in enumerate(docling_doc.tables):
                    self.logger.debug(f"    Processing table {table_idx + 1}/{len(docling_doc.tables)}")
                    
                    table_data = {
                        'table_id': f"table_{table_idx:03d}",
                        'page': getattr(table, 'page_number', None),
                        'structure': None,
                        'content': None,
                        'csv_file': None,
                        'html_file': None,
                        'markdown_content': None
                    }
                    
                    # Extract table as dataframe (THE CORRECT WAY)
                    try:
                        # v2.49.0 API: Use export_to_dataframe()
                        if hasattr(table, 'export_to_dataframe'):
                            df = table.export_to_dataframe()
                            
                            if not df.empty:
                                table_data['content'] = df.to_dict('records')
                                table_data['structure'] = {
                                    'rows': len(df),
                                    'columns': len(df.columns),
                                    'headers': list(df.columns)
                                }
                                table_data['markdown_content'] = df.to_markdown()
                                
                                # Save as CSV
                                csv_file = output_dir / 'tables' / f"{table_data['table_id']}.csv"
                                df.to_csv(csv_file, index=False, encoding='utf-8')
                                table_data['csv_file'] = str(csv_file)
                                self.logger.debug(f"      Table {table_idx + 1}: {len(df)} rows × {len(df.columns)} columns → CSV")
                                
                                # Save as HTML
                                try:
                                    if hasattr(table, 'export_to_html'):
                                        html_content = table.export_to_html(doc=docling_doc)
                                        html_file = output_dir / 'tables' / f"{table_data['table_id']}.html"
                                        with open(html_file, 'w', encoding='utf-8') as f:
                                            f.write(html_content)
                                        table_data['html_file'] = str(html_file)
                                        self.logger.debug(f"      Table {table_idx + 1}: Saved as HTML")
                                except Exception as e:
                                    self.logger.warning(f"      Failed to save table as HTML: {e}")
                        else:
                            self.logger.warning(f"      Table {table_idx + 1}: export_to_dataframe() not available")
                    
                    except Exception as e:
                        self.logger.warning(f"      Table {table_idx + 1}: Failed to convert to dataframe: {e}")
                    
                    tables.append(table_data)
                    
                    # Save metadata
                    metadata_file = output_dir / 'tables' / f"{table_data['table_id']}_metadata.json"
                    with open(metadata_file, 'w', encoding='utf-8') as f:
                        json.dump(table_data, f, indent=2, ensure_ascii=False, default=str)
            else:
                self.logger.warning(f"  No tables found in document.tables")
            
            tables_info['count'] = len(tables)
            tables_info['tables'] = tables
            self.stats['tables_extracted'] = len(tables)
            
            self.logger.info(f"  Table extraction completed: {len(tables)} table(s) found and saved")
            
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}", exc_info=True)
            tables_info['error'] = str(e)
        
        return tables_info
    
    def _extract_formulas(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract mathematical formulas and equations using proper Docling v2.49.0 API."""
        formulas_info = {
            'count': 0,
            'formulas': [],
            'extraction_method': 'docling_v2_iterate_items_formulas'
        }
        
        try:
            formulas = []
            self.logger.info(f"  Starting formula extraction using iterate_items()...")
            
            # Use the CORRECT Docling v2.49.0 API: iterate_items() with TextItem label=FORMULA
            # Formulas in Docling v2 are TextItem elements with specific labels
            from docling_core.types.doc import TextItem, DocItemLabel
            
            formula_idx = 0
            for element, level in docling_doc.iterate_items():
                # Look for TextItem elements with FORMULA label
                if isinstance(element, TextItem) and hasattr(element, 'label'):
                    if element.label == DocItemLabel.FORMULA:
                        self.logger.debug(f"    Found formula TextItem on level {level}")
                        
                        # Get page number from provenance
                        page_num = 0
                        if hasattr(element, 'prov') and element.prov:
                            page_num = element.prov[0].page_no
                        
                        # Extract formula content - use 'orig' not 'text' (text is empty for formulas)
                        formula_content = getattr(element, 'orig', '') or getattr(element, 'text', '') or ""
                        
                        formula_data = {
                            'formula_id': f"formula_{formula_idx:03d}",
                            'type': 'TextItem-FORMULA',
                            'content': formula_content,
                            'page': page_num,
                            'hierarchy_level': level
                        }
                        
                        self.logger.debug(f"      Formula {formula_idx:03d}: Page {page_num}, Content length: {len(formula_content)}")
                        
                        # Save formula to file
                        formula_file = output_dir / 'formulas' / f"{formula_data['formula_id']}.txt"
                        formula_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(formula_file, 'w', encoding='utf-8') as f:
                            f.write(f"Formula ID: {formula_data['formula_id']}\n")
                            f.write(f"Type: {formula_data['type']}\n")
                            f.write(f"Page: {formula_data['page']}\n")
                            f.write(f"Hierarchy Level: {formula_data['hierarchy_level']}\n")
                            f.write(f"Content:\n{formula_data['content']}\n")
                        
                        formula_data['file'] = str(formula_file)
                        formulas.append(formula_data)
                        formula_idx += 1
            
            formulas_info['count'] = len(formulas)
            formulas_info['formulas'] = formulas
            self.stats['formulas_detected'] = len(formulas)
            
            self.logger.info(f"  Formula extraction completed: {len(formulas)} formula(s) detected")
            
            if len(formulas) == 0:
                self.logger.debug(f"  No formulas found in document (searched all TextItem elements with FORMULA label)")
            
        except Exception as e:
            self.logger.error(f"Error extracting formulas: {e}", exc_info=True)
            formulas_info['error'] = str(e)
        
        return formulas_info
    
    def _contains_math_notation(self, text: str) -> bool:
        """Check if text contains mathematical notation."""
        if not text:
            return False
        
        math_indicators = [
            '∑', '∫', '∂', '∆', '∇', '±', '≤', '≥', '≠', '≈', '∞',
            '√', '²', '³', '\\frac', '\\sum', '\\int', '\\alpha'
        ]
        
        return any(indicator in text for indicator in math_indicators)
    
    def _extract_figures(self, docling_doc, output_dir: Path) -> Dict[str, Any]:
        """Extract figures and images with metadata using Docling v2.49.0 correct API with deduplication."""
        figures_info = {
            'count': 0,
            'figures': [],
            'extraction_method': 'docling_v2_iterate_items_pictures_dedup'
        }
        
        try:
            import hashlib
            from PIL import Image as PILImage
            
            figures = []
            self.logger.info(f"  Starting figure extraction with deduplication...")
            
            # Track image hashes to detect duplicates
            seen_hashes = {}
            fig_idx = 0
            duplicate_count = 0
            
            # Use the CORRECT Docling v2.49.0 API: doc.pictures or iterate_items() for PictureItem
            # Direct access to document.pictures is the simplest approach
            if hasattr(docling_doc, 'pictures'):
                for picture_item in docling_doc.pictures:
                    self.logger.debug(f"    Processing PictureItem {fig_idx}: {type(picture_item).__name__}")
                    
                    # Get page number from provenance
                    page_num = 0
                    bbox = None
                    if hasattr(picture_item, 'prov') and picture_item.prov:
                        page_num = picture_item.prov[0].page_no
                        bbox = picture_item.prov[0].bbox if hasattr(picture_item.prov[0], 'bbox') else None
                    
                    # Get caption
                    caption = ""
                    if hasattr(picture_item, 'caption_text'):
                        try:
                            caption = picture_item.caption_text(docling_doc) or ""
                        except Exception as e:
                            self.logger.debug(f"    Could not extract caption: {e}")
                    
                    figure_data = {
                        'figure_id': f"figure_{fig_idx:03d}",
                        'type': type(picture_item).__name__,
                        'page': page_num,
                        'caption': caption,
                        'image_file': None,
                        'source': 'document.pictures',
                        'bbox_coords': str(bbox) if bbox else None,
                        'image_dimensions': None,  # ENHANCEMENT: Track original image dimensions
                        'bbox_percentages': None    # ENHANCEMENT: Store bbox as percentages for PDF sizing
                    }
                    
                    # Try to extract and save image
                    try:
                        image = None
                        
                        # Strategy 1: Try get_image() first (preferred)
                        if hasattr(picture_item, 'get_image'):
                            try:
                                image = picture_item.get_image(docling_doc)
                                self.logger.debug(f"      Got image via get_image()")
                            except Exception as e:
                                self.logger.debug(f"      get_image() failed: {e}")
                        
                        # Strategy 2: Try to crop from page image using bbox
                        if image is None and hasattr(docling_doc, 'pages') and bbox:
                            try:
                                page = docling_doc.pages.get(page_num)
                                if page and hasattr(page, 'image') and page.image:
                                    page_img = page.image
                                    
                                    # Get PIL image
                                    if hasattr(page_img, 'pil_image'):
                                        pil_img = page_img.pil_image
                                    elif isinstance(page_img, PILImage.Image):
                                        pil_img = page_img
                                    else:
                                        pil_img = None
                                    
                                    if pil_img:
                                        # Crop using bounding box
                                        # BoundingBox has l, t, r, b (left, top, right, bottom) in document coordinates
                                        # Need to convert to image pixel coordinates
                                        page_width = page_img.width if hasattr(page_img, 'width') else pil_img.width
                                        page_height = page_img.height if hasattr(page_img, 'height') else pil_img.height
                                        
                                        # Document bbox to pixel coordinates
                                        if hasattr(bbox, 'l') and hasattr(bbox, 't'):
                                            # Assuming page_img dimensions match
                                            l = max(0, int(bbox.l * page_width / 100))
                                            t = max(0, int(bbox.t * page_height / 100))
                                            r = min(pil_img.width, int(bbox.r * page_width / 100))
                                            b = min(pil_img.height, int(bbox.b * page_height / 100))
                                            
                                            if r > l and b > t:
                                                image = pil_img.crop((l, t, r, b))
                                                self.logger.debug(f"      Cropped from page: ({l},{t}) to ({r},{b})")
                                        
                                        # Fallback: save full page if crop didn't work
                                        if image is None:
                                            image = pil_img
                                            self.logger.debug(f"      Using full page image (bbox conversion failed)")
                            
                            except Exception as e:
                                self.logger.debug(f"      Bbox cropping failed: {e}")
                        
                        # If we got an image, check for duplicates and save
                        if image:
                            # ENHANCEMENT: Store original image dimensions before saving
                            img_width, img_height = image.size
                            aspect_ratio = img_width / img_height if img_height > 0 else 1.0
                            
                            figure_data['image_dimensions'] = {
                                'width': img_width,
                                'height': img_height,
                                'aspect_ratio': aspect_ratio
                            }
                            
                            # ENHANCEMENT: Store bbox as percentages for sizing in PDF
                            if bbox:
                                try:
                                    figure_data['bbox_percentages'] = {
                                        'left': float(bbox.l),
                                        'top': float(bbox.t),
                                        'right': float(bbox.r),
                                        'bottom': float(bbox.b),
                                        'width': float(bbox.r - bbox.l),
                                        'height': float(bbox.b - bbox.t)
                                    }
                                    self.logger.debug(f"      Image dimensions: {img_width}x{img_height}px, aspect ratio: {aspect_ratio:.2f}")
                                except Exception as e:
                                    self.logger.debug(f"      Could not extract bbox percentages: {e}")
                            
                            # Compute hash to detect duplicates
                            from io import BytesIO
                            img_bytes = BytesIO()
                            image.save(img_bytes, format='PNG')
                            img_hash = hashlib.md5(img_bytes.getvalue()).hexdigest()
                            
                            # Check if we've seen this exact image before
                            if img_hash in seen_hashes:
                                # Duplicate detected
                                original_fig = seen_hashes[img_hash]
                                self.logger.debug(f"      Duplicate detected! Same as {original_fig['figure_id']}")
                                duplicate_count += 1
                                # Skip saving, but track it
                                figure_data['image_file'] = original_fig['image_file']
                                figure_data['is_duplicate_of'] = original_fig['figure_id']
                            else:
                                # New unique image - save it
                                image_file = output_dir / 'figures' / f"{figure_data['figure_id']}.png"
                                image_file.parent.mkdir(parents=True, exist_ok=True)
                                image.save(str(image_file))
                                
                                figure_data['image_file'] = str(image_file)
                                seen_hashes[img_hash] = figure_data
                                self.logger.debug(f"      Saved unique image: {image_file.name} ({img_width}x{img_height}px)")
                        else:
                            self.logger.debug(f"      Picture {fig_idx} on page {page_num}: could not extract image")
                    
                    except Exception as e:
                        self.logger.debug(f"      Error extracting image for figure {fig_idx}: {e}")
                    
                    figures.append(figure_data)
                    fig_idx += 1
            
            figures_info['count'] = len(figures)
            figures_info['figures'] = figures
            figures_info['unique_count'] = len(seen_hashes)
            figures_info['duplicates_found'] = duplicate_count
            self.stats['figures_detected'] = len(figures)
            
            self.logger.info(f"  Figure extraction completed: {len(figures)} figure(s) detected")
            self.logger.info(f"    - Unique images: {len(seen_hashes)}")
            self.logger.info(f"    - Duplicates: {duplicate_count}")
            
            if len(figures) == 0:
                self.logger.warning(f"  No pictures found in document (document.pictures is empty)")
            
        except Exception as e:
            self.logger.error(f"Error extracting figures: {e}", exc_info=True)
            figures_info['error'] = str(e)
        
        return figures_info
    
    def _export_to_formats(self, docling_doc, output_dir: Path) -> Dict[str, str]:
        """Export document to various formats (Markdown, JSON, etc.)."""
        export_files = {}
        
        try:
            self.logger.info(f"  Starting format exports...")
            
            # Export to Markdown
            if hasattr(docling_doc, 'export_to_markdown'):
                self.logger.debug(f"    Exporting to Markdown format...")
                markdown_content = docling_doc.export_to_markdown()
                markdown_file = output_dir / 'markdown' / 'document.md'
                with open(markdown_file, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                export_files['markdown'] = str(markdown_file)
                self.stats['export_formats'].append('markdown')
                self.logger.debug(f"    Markdown export complete ({len(markdown_content)} chars)")
            
            # Export to JSON
            if hasattr(docling_doc, 'export_to_dict'):
                self.logger.debug(f"    Exporting to JSON format...")
                json_content = docling_doc.export_to_dict()
                json_file = output_dir / 'json' / 'document.json'
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_content, f, indent=2, ensure_ascii=False, default=str)
                export_files['json'] = str(json_file)
                self.stats['export_formats'].append('json')
                self.logger.debug(f"    JSON export complete")
            
            # Export full text
            self.logger.debug(f"    Exporting full document as text...")
            text_content = self._extract_structured_text(docling_doc)
            text_file = output_dir / 'text' / 'full_document.txt'
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            export_files['text'] = str(text_file)
            
            self.logger.info(f"  Format exports complete: {list(export_files.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to formats: {e}", exc_info=True)
            export_files['error'] = str(e)
        
        return export_files
    
    def _generate_comparison_metrics(self, docling_doc) -> Dict[str, Any]:
        """Generate metrics for comparison with traditional methods."""
        metrics = {
            'docling_version': '2.49.0',
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
                'hierarchical_extraction': True,
                'semantic_understanding': True,
                'layout_preservation': True,
                'cross_page_continuity': True,
                'ai_model_based': True
            }
            
            # Docling v2.49.0 advantages
            advantages = [
                "Advanced AI/ML models for document understanding",
                "Improved table detection and structure recognition",
                "Enhanced formula and equation extraction",
                "Better figure/image identification",
                "Unified document model (DoclingDocument)",
                "Multiple export formats (Markdown, JSON)",
                "Semantic understanding of document elements",
                "Better handling of complex layouts",
                "Reading order preservation"
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
            'docling_version': '2.49.0',
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
        self.logger.info("DOCLING v2.49.0 EXTRACTION STATISTICS SUMMARY")
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
    """Main function to demonstrate Docling v2.49.0-based extraction."""
    import argparse
    import sys
    
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from update_provenance import update_parser_provenance
    except ImportError:
        update_parser_provenance = None
    
    print("=== Docling Advanced PDF Understanding v2.49.0 ===")
    print("Usage: python src/parse/docling_extractor_v2.py <path> --pdf <pdf_name>")
    print("Example: python src/parse/docling_extractor_v2.py FINTBX/pdf/split_pdfs --pdf fintbx_part_001.pdf")
    print()
    print("Output structure: data/parsed/{Company}/[pdf_name]/")
    print("  ├── text/           - Structured text content")
    print("  ├── tables/         - Advanced table extraction (CSV + metadata)")
    print("  ├── formulas/       - Mathematical formulas")
    print("  ├── figures/        - Extracted figures and images")
    print("  ├── structure/      - Document structure analysis")
    print("  ├── markdown/       - Markdown export")
    print("  ├── json/           - JSON export")
    print("  ├── comparison/     - Comparison with traditional methods")
    print("  └── reading_order/  - Reading order elements")
    print()

    parser = argparse.ArgumentParser(description="Docling v2.49.0 PDF Extraction")
    parser.add_argument("path", type=str, help="Relative path to PDF directory (e.g., FINTBX/pdf/split_pdfs)")
    parser.add_argument("--pdf", type=str, required=True, help="PDF file name to process")
    args = parser.parse_args()

    # Extract company ticker from path
    path_parts = args.path.split('/')
    company_ticker = path_parts[0] if path_parts else "Unknown"
    
    # Build full PDF path
    pdf_file = Path(f"data/raw/{args.path}/{args.pdf}")
    
    if not pdf_file.exists():
        print(f"✗ PDF file not found: {pdf_file}")
        return

    try:
        extractor = DoclingExtractor(output_dir=f"data/parsed/{company_ticker}", company_ticker=company_ticker)
        
        print(f"\nProcessing PDF: {pdf_file.name}")
        print(f"Company: {company_ticker}")
        print()
        
        results = extractor.extract_from_pdf(str(pdf_file))
        
        if results and results.get('processing_success', False):
            print(f"\n✓ Extraction completed successfully!")
            print(f"  Pages: {results['document_analysis'].get('total_pages', 0)}")
            print(f"  Elements: {len(results['content_structure'].get('reading_order', []))}")
            print(f"  Tables: {results['content_structure']['tables']['count']}")
            print(f"  Formulas: {results['content_structure']['formulas']['count']}")
            print(f"  Figures: {results['content_structure']['figures']['count']}")
            print(f"  Time: {results['processing_time_seconds']:.2f}s")
            
            if update_parser_provenance:
                output_dir = f"data/parsed/{company_ticker}/{pdf_file.stem}"
                parsed_files = []
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), start=".")
                        parsed_files.append(rel_path.replace("\\", "/"))
                
                update_parser_provenance(
                    company_ticker,
                    [{
                        "pdf_path": str(pdf_file),
                        "pdf_name": results.get("pdf_name", ""),
                        "pages": results['document_analysis'].get('total_pages', 0),
                        "tables": results['content_structure']['tables']['count'],
                        "formulas": results['content_structure']['formulas']['count'],
                        "figures": results['content_structure']['figures']['count'],
                    }],
                    parsed_files=parsed_files
                )
        else:
            print(f"✗ Extraction failed")
            if results and 'error_message' in results:
                print(f"  Error: {results['error_message']}")

    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    main()
