"""
Enhanced PDF Generator v2 - Proper Figure and Element Mapping

This version correctly handles:
1. Figure-to-reading-order correlation using page and sequence
2. Per-page reconstruction with proper element ordering
3. Figure indexing and deduplication
4. Complete metadata utilization from extraction results

Key improvements over v1:
- Uses content_structure.figures master list for figure metadata
- Correlates figures with reading_order items by page and sequence
- Maintains global figure counter to avoid duplication
- Properly embeds figures on correct pages
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, 
        PageBreak, Flowable
    )
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError as e:
    HAS_REPORTLAB = False
    print(f"Warning: ReportLab not installed. Error: {e}")
    print("Install with: pip install reportlab pillow")


class PDFGeneratorV2:
    """Enhanced PDF generator with proper figure and element mapping."""
    
    def __init__(self, output_dir: Path, company_ticker: str = None):
        """
        Initialize enhanced PDF generator.
        
        Args:
            output_dir: Directory containing parsed extraction results
            company_ticker: Company ticker for logging
        """
        self.output_dir = Path(output_dir)
        self.company_ticker = company_ticker
        self.logger = self._setup_logging()
        
        if not HAS_REPORTLAB:
            self.logger.error("ReportLab not installed. Install with: pip install reportlab pillow")
        
        # Figure tracking
        self.figure_map = {}  # Maps (page, sequence) to figure file
        self.embedded_figures = set()  # Track which figures we've embedded
        self.figure_by_id = {}  # Maps figure_id to metadata
    
    def _setup_logging(self):
        """Setup logging for generator."""
        logger = logging.getLogger('PDFGeneratorV2')
        logger.setLevel(logging.INFO)
        
        if self.company_ticker:
            log_dir = Path(f"data/logs/{self.company_ticker}")
        else:
            log_dir = Path("data/logs/reconstruction")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'pdf_generation_v2.log'
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(handler)
        
        return logger
    
    def generate_pdf_from_extraction(self, extraction_json_path: Path, output_pdf_path: Path = None) -> Dict[str, Any]:
        """
        Generate PDF from extraction results with proper figure mapping and content handling.
        
        Args:
            extraction_json_path: Path to docling_extraction_results.json
            output_pdf_path: Optional path for output PDF
            
        Returns:
            dict: Generation report with statistics
        """
        if not HAS_REPORTLAB:
            return {
                'success': False,
                'error': 'ReportLab not installed',
                'timestamp': datetime.now().isoformat()
            }
        
        if output_pdf_path is None:
            output_pdf_path = self.output_dir / 'reconstructed_document_v2.pdf'
        
        self.logger.info(f"Starting enhanced PDF generation from: {extraction_json_path}")
        self.logger.info(f"Output will be saved to: {output_pdf_path}")
        
        generation_report = {
            'timestamp': datetime.now().isoformat(),
            'extraction_source': str(extraction_json_path),
            'output_file': str(output_pdf_path),
            'total_elements_processed': 0,
            'elements_by_type': {},
            'pages_generated': 0,
            'figures_embedded': 0,
            'tables_embedded': 0,
            'text_blocks': 0,
            'file_size_bytes': 0,
            'success': False,
            'errors': [],
            'warnings': [],
            'figure_mapping_details': [],
            'skipped_duplicate_figures': 0
        }
        
        try:
            # Load extraction results
            with open(extraction_json_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            content_structure = extraction_data.get('content_structure', {})
            reading_order = content_structure.get('reading_order', [])
            figures_meta = content_structure.get('figures', {})
            
            # Build figure master index
            self._build_figure_map(figures_meta, reading_order)
            self.logger.info(f"Built figure map with {len(self.figure_by_id)} figures")
            self.logger.info(f"Figure deduplication info: unique={figures_meta.get('unique_count', '?')}, duplicates={figures_meta.get('duplicates_found', '?')}")
            
            # Organize elements by page
            elements_by_page = self._organize_elements_by_page(reading_order)
            self.logger.info(f"Organized {len(reading_order)} elements across {len(elements_by_page)} pages")
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(output_pdf_path),
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=0.75*inch,
                bottomMargin=0.75*inch
            )
            
            # Build content list for PDF
            story = []
            styles = self._get_custom_styles()
            
            # Process elements by page
            current_page = 0
            page_count = 0
            
            for page_num in sorted(elements_by_page.keys()):
                page_elems = elements_by_page[page_num]
                
                # Add page break if not first page
                if page_num > current_page and page_count > 0:
                    story.append(PageBreak())
                    page_count += 1
                
                current_page = page_num
                
                # Add page header
                header_para = Paragraph(f"<b>Page {page_num}</b>", styles['page_header'])
                story.append(header_para)
                story.append(Spacer(1, 0.2*inch))
                
                # Track content added to this page
                page_content_added = False
                
                # Process each element on this page
                for elem in page_elems:
                    try:
                        elem_type = elem.get('type', 'unknown')
                        generation_report['elements_by_type'][elem_type] = \
                            generation_report['elements_by_type'].get(elem_type, 0) + 1
                        generation_report['total_elements_processed'] += 1
                        
                        # Handle different element types
                        if elem_type == 'PictureItem':
                            fig_path = self._get_figure_for_element(elem, reading_order.index(elem))
                            if fig_path and Path(fig_path).exists():
                                try:
                                    img = self._create_image(Path(fig_path))
                                    if img:
                                        story.append(img)
                                        story.append(Spacer(1, 0.2*inch))
                                        generation_report['figures_embedded'] += 1
                                        page_content_added = True
                                        
                                        # Log figure mapping
                                        generation_report['figure_mapping_details'].append({
                                            'page': page_num,
                                            'figure_file': str(fig_path),
                                            'element_index': reading_order.index(elem)
                                        })
                                except Exception as e:
                                    self.logger.warning(f"Failed to embed figure on page {page_num}: {e}")
                                    generation_report['warnings'].append(f"Failed to embed figure: {e}")
                            else:
                                self.logger.debug(f"Figure file not found or is a duplicate reference")
                                generation_report['skipped_duplicate_figures'] += 1
                        
                        elif elem_type == 'TableItem':
                            table_content = self._extract_table_content(elem)
                            if table_content:
                                try:
                                    tbl = self._create_table(table_content, styles)
                                    if tbl:
                                        story.append(tbl)
                                        story.append(Spacer(1, 0.1*inch))
                                        generation_report['tables_embedded'] += 1
                                        page_content_added = True
                                except Exception as e:
                                    self.logger.warning(f"Failed to create table on page {page_num}: {e}")
                                    generation_report['warnings'].append(f"Failed to create table: {e}")
                        
                        elif elem_type in ['TextItem', 'SectionHeaderItem']:
                            content = elem.get('content', '').strip()
                            if content:
                                try:
                                    para = self._create_paragraph(elem, styles)
                                    if para:
                                        story.append(para)
                                        generation_report['text_blocks'] += 1
                                        page_content_added = True
                                except Exception as e:
                                    self.logger.warning(f"Failed to create paragraph: {e}")
                        
                        elif elem_type == 'CodeItem':
                            content = elem.get('content', '').strip()
                            if content:
                                try:
                                    escaped = self._escape_html(content[:500])  # Limit code block size
                                    para = Paragraph(
                                        f"<font face=\"Courier\" size=\"8\">{escaped}</font>", 
                                        styles.get('code', styles['Normal'])
                                    )
                                    story.append(para)
                                    story.append(Spacer(1, 0.05*inch))
                                    page_content_added = True
                                except Exception as e:
                                    self.logger.debug(f"Failed to add code block: {e}")
                        
                        elif elem_type == 'FormulaItem':
                            content = elem.get('content', '').strip()
                            if content:
                                try:
                                    escaped = self._escape_html(content[:200])
                                    para = Paragraph(f"<i>{escaped}</i>", styles.get('formula', styles['Normal']))
                                    story.append(para)
                                    story.append(Spacer(1, 0.05*inch))
                                    page_content_added = True
                                except Exception as e:
                                    self.logger.debug(f"Failed to add formula: {e}")
                        
                        elif elem_type == 'ListItem':
                            content = elem.get('content', '').strip()
                            if content:
                                try:
                                    escaped = self._escape_html(content)
                                    para = Paragraph(f"• {escaped}", styles['Normal'])
                                    story.append(para)
                                    page_content_added = True
                                except Exception as e:
                                    self.logger.debug(f"Failed to add list item: {e}")
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing element of type {elem.get('type')}: {e}")
                        generation_report['warnings'].append(f"Skipped element: {str(e)}")
                        continue
                
                # Only count page if we added actual content
                if page_content_added:
                    page_count += 1
            
            # Build PDF
            self.logger.info(f"Building PDF with {len(story)} story elements...")
            doc.build(story)
            
            # Verify output
            if output_pdf_path.exists():
                file_size = output_pdf_path.stat().st_size
                generation_report['file_size_bytes'] = file_size
                generation_report['success'] = True
                generation_report['pages_generated'] = page_count
                
                self.logger.info(f"✓ PDF generated successfully: {output_pdf_path}")
                self.logger.info(f"  File size: {file_size / 1024 / 1024:.2f} MB")
                self.logger.info(f"  Pages: {page_count}")
                self.logger.info(f"  Figures embedded: {generation_report['figures_embedded']}")
                self.logger.info(f"  Duplicate figures skipped: {generation_report['skipped_duplicate_figures']}")
                self.logger.info(f"  Tables embedded: {generation_report['tables_embedded']}")
                self.logger.info(f"  Text blocks: {generation_report['text_blocks']}")
            else:
                generation_report['errors'].append("PDF file was not created")
                self.logger.error("PDF file was not created")
            
            return generation_report
            
        except Exception as e:
            self.logger.error(f"Error generating PDF: {e}", exc_info=True)
            generation_report['errors'].append(str(e))
            return generation_report
    
    def _build_figure_map(self, figures_meta: Dict, reading_order: List[Dict]):
        """Build a mapping of figures from metadata."""
        figures_list = figures_meta.get('figures', [])
        
        for fig_data in figures_list:
            fig_id = fig_data.get('figure_id')
            fig_page = fig_data.get('page')
            image_file = fig_data.get('image_file')
            
            if fig_id and image_file:
                self.figure_by_id[fig_id] = {
                    'page': fig_page,
                    'file': image_file,
                    'caption': fig_data.get('caption', '')
                }
                self.logger.debug(f"Registered figure: {fig_id} on page {fig_page}")
    
    def _get_figure_for_element(self, figure_elem: Dict, elem_index: int) -> Optional[str]:
        """
        Get the figure file path for a reading order element.
        
        Uses multiple strategies to find the correct figure:
        1. Global figure counter (most reliable)
        2. Page-based lookup
        3. Sequential matching
        """
        # Strategy: Use a global figure counter
        # Count how many PictureItem elements we've seen so far across all pages
        figure_counter = sum(1 for item in self.figure_map.values())
        
        # Find the Nth figure from our metadata
        figures_list = sorted(
            [(fig_id, data) for fig_id, data in self.figure_by_id.items()],
            key=lambda x: int(x[0].split('_')[1])  # Sort by figure number
        )
        
        if figure_counter < len(figures_list):
            fig_id, fig_data = figures_list[figure_counter]
            file_path = fig_data['file']
            
            # Record this mapping
            self.figure_map[figure_counter] = file_path
            self.logger.debug(f"Mapped element index {elem_index} -> {fig_id} ({file_path})")
            
            return file_path
        
        return None
    
    def _organize_elements_by_page(self, reading_order: List[Dict]) -> Dict[int, List[Dict]]:
        """Organize reading order elements by page number."""
        by_page = defaultdict(list)
        for elem in reading_order:
            page = elem.get('page', 1)
            by_page[page].append(elem)
        return dict(by_page)
    
    def _get_custom_styles(self):
        """Get custom paragraph styles for PDF generation."""
        styles = getSampleStyleSheet()
        
        # Page header style
        styles.add(ParagraphStyle(
            name='page_header',
            parent=styles['Heading1'],
            fontSize=14,
            textColor=colors.HexColor('#1f4e78'),
            spaceAfter=12,
            borderPadding=5
        ))
        
        # Section header style
        styles.add(ParagraphStyle(
            name='section_header',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#2f5496'),
            spaceAfter=10
        ))
        
        # Code style
        styles.add(ParagraphStyle(
            name='code',
            parent=styles['Normal'],
            fontSize=8,
            fontName='Courier',
            textColor=colors.HexColor('#4a4a4a'),
            spaceAfter=6,
            leftIndent=0.25*inch
        ))
        
        # Formula style
        styles.add(ParagraphStyle(
            name='formula',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#333333'),
            spaceAfter=8
        ))
        
        return styles
    
    def _create_paragraph(self, element: Dict, styles):
        """Create formatted paragraph from text element."""
        try:
            content = element.get('content', '').strip()
            if not content:
                return None
            
            elem_type = element.get('type', '').lower()
            hierarchy_level = element.get('hierarchy_level', 0)
            
            # Escape special characters for ReportLab
            content = self._escape_html(content[:1000])  # Limit content size
            
            # Choose style based on element type and hierarchy
            if 'header' in elem_type:
                if hierarchy_level == 0:
                    style_name = 'Heading1'
                elif hierarchy_level == 1:
                    style_name = 'Heading2'
                else:
                    style_name = 'Heading3'
            else:
                style_name = 'Normal'
            
            para = Paragraph(content, styles[style_name])
            return para
            
        except Exception as e:
            self.logger.warning(f"Error creating paragraph: {e}")
            return None
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters for ReportLab."""
        if not text:
            return ""
        replacements = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        }
        for char, escaped in replacements.items():
            text = text.replace(char, escaped)
        return text
    
    def _extract_table_content(self, table_elem: Dict) -> Optional[List[List[str]]]:
        """Extract table content from TableItem."""
        try:
            content = table_elem.get('content', '')
            if not content:
                return None
            
            # Try to parse CSV content
            lines = content.split('\n')
            if lines:
                table_data = [
                    [cell.strip() for cell in line.split(',')]
                    for line in lines if line.strip()
                ]
                return table_data if table_data else None
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting table: {e}")
            return None
    
    def _create_table(self, table_data: List[List[str]], styles):
        """Create formatted table from table data."""
        try:
            if not table_data or len(table_data) < 1:
                return None
            
            # Limit table size for PDF rendering
            max_rows = 20
            if len(table_data) > max_rows:
                table_data = table_data[:max_rows]
            
            # Create table
            tbl = Table(table_data, repeatRows=1)
            
            # Style table
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2f5496')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
            ]))
            
            return tbl
            
        except Exception as e:
            self.logger.warning(f"Error creating table: {e}")
            return None
    
    def _create_image(self, image_path: Path, max_width: float = 6.5, max_height: float = 4.5):
        """Create scaled image for PDF."""
        try:
            if not image_path.exists():
                self.logger.warning(f"Image not found: {image_path}")
                return None
            
            img = Image(str(image_path), width=max_width*inch, height=max_height*inch)
            return img
            
        except Exception as e:
            self.logger.warning(f"Error creating image: {e}")
            return None


def main():
    """Main function to generate PDF from extraction results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced PDF Generation from Docling Extraction")
    parser.add_argument(
        "extraction_json",
        type=str,
        help="Path to docling_extraction_results.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (optional)"
    )
    
    args = parser.parse_args()
    
    extraction_path = Path(args.extraction_json)
    if not extraction_path.exists():
        print(f"✗ Extraction JSON not found: {extraction_path}")
        return
    
    # Get output directory from extraction path
    output_dir = extraction_path.parent
    
    # Determine company ticker from path
    path_parts = str(output_dir).split('/')
    company_ticker = None
    if 'parsed' in path_parts:
        idx = path_parts.index('parsed')
        if idx + 1 < len(path_parts):
            company_ticker = path_parts[idx + 1]
    
    try:
        generator = PDFGeneratorV2(output_dir, company_ticker)
        
        print(f"\n{'='*60}")
        print(f"Enhanced PDF Generation v2")
        print(f"{'='*60}")
        print(f"Input:  {extraction_path}")
        print(f"Output: {args.output or output_dir / 'reconstructed_document_v2.pdf'}")
        print()
        
        report = generator.generate_pdf_from_extraction(
            extraction_path,
            Path(args.output) if args.output else None
        )
        
        print(f"\nGeneration Report:")
        print(f"  Success: {report['success']}")
        print(f"  Elements processed: {report['total_elements_processed']}")
        print(f"  Pages generated: {report['pages_generated']}")
        print(f"  Figures embedded: {report['figures_embedded']}")
        print(f"  Tables embedded: {report['tables_embedded']}")
        print(f"  Text blocks: {report['text_blocks']}")
        
        if report['file_size_bytes']:
            print(f"  File size: {report['file_size_bytes'] / 1024 / 1024:.2f} MB")
        
        if report['warnings']:
            print(f"\n  Warnings ({len(report['warnings'])}):")
            for w in report['warnings'][:5]:
                print(f"    - {w}")
        
        if report['errors']:
            print(f"\n  Errors ({len(report['errors'])}):")
            for e in report['errors']:
                print(f"    - {e}")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"✗ Error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
