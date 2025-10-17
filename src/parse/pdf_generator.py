"""
PDF Generation from Parsed Elements

Reconstructs a PDF file from extracted Docling elements with:
1. Text with proper formatting and hierarchy
2. Tables with structure preserved
3. Figures embedded at correct positions
4. Reading order maintained
5. Page layout preserved from provenance data
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from io import BytesIO
import os

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError as e:
    HAS_REPORTLAB = False
    print(f"Warning: ReportLab not installed. Error: {e}")
    print("Install with: pip install reportlab pillow")


class PDFGenerator:
    """Generates PDF from parsed document elements."""
    
    def __init__(self, output_dir: Path, company_ticker: str = None):
        """
        Initialize PDF generator.
        
        Args:
            output_dir: Directory containing parsed extraction results
            company_ticker: Company ticker for logging
        """
        self.output_dir = Path(output_dir)
        self.company_ticker = company_ticker
        self.logger = self._setup_logging()
        
        if not HAS_REPORTLAB:
            self.logger.error("ReportLab not installed. Install with: pip install reportlab pillow")
    
    def _setup_logging(self):
        """Setup logging for generator."""
        logger = logging.getLogger('PDFGenerator')
        logger.setLevel(logging.INFO)
        
        if self.company_ticker:
            log_dir = Path(f"data/logs/{self.company_ticker}")
        else:
            log_dir = Path("data/logs/reconstruction")
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / 'pdf_generation.log'
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
        Generate PDF from extraction results.
        
        Args:
            extraction_json_path: Path to docling_extraction_results.json
            output_pdf_path: Optional path for output PDF (default: reconstructed.pdf)
            
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
            output_pdf_path = self.output_dir / 'reconstructed_document.pdf'
        
        self.logger.info(f"Starting PDF generation from: {extraction_json_path}")
        
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
            'warnings': []
        }
        
        try:
            # Load extraction results
            with open(extraction_json_path, 'r', encoding='utf-8') as f:
                extraction_data = json.load(f)
            
            content_structure = extraction_data.get('content_structure', {})
            reading_order = content_structure.get('reading_order', [])
            
            # Organize elements by page
            elements_by_page = self._organize_elements_by_page(reading_order)
            
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
            page_elements = 0
            
            for page_num in sorted(elements_by_page.keys()):
                page_elems = elements_by_page[page_num]
                
                if page_num > current_page and page_elements > 0:
                    # Add page break between pages
                    story.append(PageBreak())
                    generation_report['pages_generated'] += 1
                    page_elements = 0
                
                current_page = page_num
                
                # Add page header
                story.append(Paragraph(f"<b>Page {page_num}</b>", styles['page_header']))
                story.append(Spacer(1, 0.2*inch))
                
                # Process elements on this page
                for elem in page_elems:
                    try:
                        elem_type = elem.get('type', 'unknown')
                        generation_report['elements_by_type'][elem_type] = \
                            generation_report['elements_by_type'].get(elem_type, 0) + 1
                        generation_report['total_elements_processed'] += 1
                        
                        if 'text' in elem_type.lower():
                            content = elem.get('content', '')
                            if content:
                                para = self._create_paragraph(elem, styles)
                                if para:
                                    story.append(para)
                                    generation_report['text_blocks'] += 1
                        
                        elif elem_type == 'TableItem':
                            table_content = self._extract_table_content(elem)
                            if table_content:
                                tbl = self._create_table(table_content, styles)
                                if tbl:
                                    story.append(tbl)
                                    story.append(Spacer(1, 0.1*inch))
                                    generation_report['tables_embedded'] += 1
                        
                        elif elem_type == 'PictureItem':
                            fig_path = self._find_figure_file(elem)
                            if fig_path and fig_path.exists():
                                img = self._create_image(fig_path)
                                if img:
                                    story.append(img)
                                    story.append(Spacer(1, 0.1*inch))
                                    generation_report['figures_embedded'] += 1
                        
                        elif elem_type == 'CodeItem':
                            content = elem.get('content', '')
                            if content:
                                para = Paragraph(f"<font face=\"Courier\" size=\"9\">{content}</font>", 
                                               styles['code'])
                                story.append(para)
                        
                        elif elem_type == 'FormulaItem':
                            content = elem.get('content', '')
                            if content:
                                para = Paragraph(f"<i>{content}</i>", styles['formula'])
                                story.append(para)
                        
                        page_elements += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing element {elem.get('id')}: {e}")
                        generation_report['warnings'].append(f"Skipped element: {str(e)}")
                        continue
            
            # Build PDF
            self.logger.info(f"Building PDF with {len(story)} story elements...")
            doc.build(story)
            
            # Get file size
            if output_pdf_path.exists():
                generation_report['file_size_bytes'] = output_pdf_path.stat().st_size
                generation_report['success'] = True
                self.logger.info(f"PDF generated successfully: {output_pdf_path}")
                self.logger.info(f"File size: {generation_report['file_size_bytes'] / 1024 / 1024:.2f} MB")
            
            return generation_report
            
        except Exception as e:
            self.logger.error(f"Error generating PDF: {e}", exc_info=True)
            generation_report['errors'].append(str(e))
            return generation_report
    
    def _organize_elements_by_page(self, reading_order: List[Dict]) -> Dict[int, List[Dict]]:
        """Organize reading order elements by page number."""
        by_page = {}
        for elem in reading_order:
            page = elem.get('page', 1)
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(elem)
        return by_page
    
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
            spaceAfter=6
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
            content = self._escape_html(content)
            
            # Choose style based on element type and hierarchy
            if 'header' in elem_type:
                if hierarchy_level == 1:
                    style_name = 'Heading1'
                elif hierarchy_level == 2:
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
        """Escape HTML special characters."""
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
    
    def _extract_table_content(self, table_elem: Dict) -> List[List[str]]:
        """Extract table content from TableItem."""
        try:
            content = table_elem.get('content', '')
            
            # Try to parse CSV content
            lines = content.split('\n')
            if lines:
                table_data = [line.split(',') for line in lines if line.strip()]
                return table_data if table_data else None
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error extracting table: {e}")
            return None
    
    def _create_table(self, table_data: List[List[str]], styles):
        """Create formatted table from table data."""
        try:
            if not table_data or len(table_data) < 2:
                return None
            
            # Create table
            tbl = Table(table_data, repeatRows=1)
            
            # Style table
            tbl.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2f5496')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
            ]))
            
            return tbl
            
        except Exception as e:
            self.logger.warning(f"Error creating table: {e}")
            return None
    
    def _find_figure_file(self, figure_elem: Dict) -> Path:
        """Find figure file from parsed figures directory."""
        try:
            figure_id = figure_elem.get('id', '')
            figures_dir = self.output_dir / 'figures'
            
            if figures_dir.exists():
                # Search for matching figure file
                for fig_file in figures_dir.glob('*.png'):
                    if figure_id in fig_file.name or fig_file.stem.endswith(figure_id.split('_')[-1]):
                        return fig_file
                
                # If no exact match, get first available or search by index
                fig_num = figure_elem.get('index', 0)
                fig_files = sorted(figures_dir.glob('*.png'))
                if fig_num < len(fig_files):
                    return fig_files[fig_num]
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding figure: {e}")
            return None
    
    def _create_image(self, image_path: Path, max_width: float = 6.5, max_height: float = 4.5):
        """Create scaled image for PDF."""
        try:
            img = Image(str(image_path), width=max_width*inch, height=max_height*inch)
            return img
            
        except Exception as e:
            self.logger.warning(f"Error creating image: {e}")
            return None


def generate_reconstructed_pdf(extraction_dir: Path, company_ticker: str = None, 
                               output_path: Path = None) -> Dict[str, Any]:
    """
    Main function to generate PDF from extracted elements.
    
    Args:
        extraction_dir: Directory containing parsed extraction results
        company_ticker: Company ticker for logging
        output_path: Optional custom output path for PDF
        
    Returns:
        dict: Generation report with statistics
    """
    generator = PDFGenerator(extraction_dir, company_ticker)
    
    extraction_json = extraction_dir / 'docling_extraction_results.json'
    
    if not extraction_json.exists():
        raise FileNotFoundError(f"Extraction results not found: {extraction_json}")
    
    if output_path is None:
        output_path = extraction_dir / 'reconstructed_document.pdf'
    
    report = generator.generate_pdf_from_extraction(extraction_json, output_path)
    
    return report


if __name__ == "__main__":
    # Example usage
    extraction_dir = Path("data/parsed/FINTBX/fintbx_part_001")
    
    print("=== PDF Generation from Extracted Elements ===\n")
    
    try:
        report = generate_reconstructed_pdf(extraction_dir, "FINTBX")
        
        print(f"Status: {'✓ SUCCESS' if report['success'] else '✗ FAILED'}")
        print(f"Output: {report.get('output_file', 'N/A')}")
        print(f"Total elements processed: {report.get('total_elements_processed', 0)}")
        print(f"Text blocks: {report.get('text_blocks', 0)}")
        print(f"Tables embedded: {report.get('tables_embedded', 0)}")
        print(f"Figures embedded: {report.get('figures_embedded', 0)}")
        print(f"Pages generated: {report.get('pages_generated', 0)}")
        file_size = report.get('file_size_bytes', 0)
        if file_size > 0:
            print(f"File size: {file_size / 1024 / 1024:.2f} MB")
        
        if report.get('errors'):
            print(f"\nErrors ({len(report['errors'])}):")
            for err in report['errors'][:5]:
                print(f"  - {err}")
        
        if report.get('warnings'):
            print(f"\nWarnings ({len(report['warnings'])}):")
            for warn in report['warnings'][:5]:
                print(f"  - {warn}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
