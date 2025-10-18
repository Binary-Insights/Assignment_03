"""
Spacing and Positioning Extractor for PDF Documents

This module enhances the basic extraction with spatial information:
- Bounding box coordinates for all elements
- Spacing/indentation calculations
- Element positioning information
- Alignment detection

This allows the PDF reconstruction to preserve the original layout
and spacing from the source document.

Author: Enhancement Module
Version: 1.0
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging


@dataclass
class BoundingBox:
    """Represents element position on page."""
    left: float      # Left coordinate (0-100%)
    top: float       # Top coordinate (0-100%)
    right: float     # Right coordinate (0-100%)
    bottom: float    # Bottom coordinate (0-100%)
    
    @property
    def width(self) -> float:
        """Calculate width as percentage of page."""
        return self.right - self.left
    
    @property
    def height(self) -> float:
        """Calculate height as percentage of page."""
        return self.bottom - self.top
    
    @property
    def center_x(self) -> float:
        """Horizontal center position."""
        return (self.left + self.right) / 2
    
    @property
    def center_y(self) -> float:
        """Vertical center position."""
        return (self.top + self.bottom) / 2
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'left': self.left,
            'top': self.top,
            'right': self.right,
            'bottom': self.bottom,
            'width': self.width,
            'height': self.height,
            'center_x': self.center_x,
            'center_y': self.center_y,
        }
    
    @staticmethod
    def from_docling(bbox) -> Optional['BoundingBox']:
        """Create from Docling bbox object."""
        if bbox is None:
            return None
        
        try:
            if hasattr(bbox, 'l') and hasattr(bbox, 't'):
                return BoundingBox(
                    left=float(bbox.l),
                    top=float(bbox.t),
                    right=float(bbox.r),
                    bottom=float(bbox.b)
                )
        except (AttributeError, ValueError, TypeError):
            pass
        
        return None


@dataclass
class ElementSpacing:
    """Represents spacing around an element."""
    margin_top: Optional[float] = None      # Points above
    margin_bottom: Optional[float] = None   # Points below
    margin_left: Optional[float] = None     # Points left
    margin_right: Optional[float] = None    # Points right
    line_height: Optional[float] = None     # For text elements
    text_indent: Optional[float] = None     # First line indent
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary."""
        return {
            'margin_top': self.margin_top,
            'margin_bottom': self.margin_bottom,
            'margin_left': self.margin_left,
            'margin_right': self.margin_right,
            'line_height': self.line_height,
            'text_indent': self.text_indent,
        }


class SpacingCalculator:
    """Calculate spacing between elements on a page."""
    
    def __init__(self, page_height_percent: float = 100.0):
        """
        Initialize calculator.
        
        Args:
            page_height_percent: Height of page as 100% reference
        """
        self.page_height = page_height_percent
        self.logger = logging.getLogger('SpacingCalculator')
    
    def calculate_spacing_between(self, prev_bbox: Optional[BoundingBox], 
                                  curr_bbox: Optional[BoundingBox]) -> Optional[float]:
        """
        Calculate vertical spacing between two elements.
        
        Args:
            prev_bbox: Previous element's bounding box
            curr_bbox: Current element's bounding box
            
        Returns:
            Spacing in percentage points, or None if calculation not possible
        """
        if not prev_bbox or not curr_bbox:
            return None
        
        # Distance from bottom of previous to top of current
        spacing_percent = curr_bbox.top - prev_bbox.bottom
        
        # Return as percentage (can be negative if overlapping)
        return max(0, spacing_percent)  # Ensure non-negative
    
    def infer_indentation(self, bbox: BoundingBox, 
                         page_left_margin: float = 5.0) -> Optional[float]:
        """
        Infer left indentation from bbox.
        
        Args:
            bbox: Element's bounding box
            page_left_margin: Expected page left margin in %
            
        Returns:
            Indentation level (0 = no indent, >0 = indented)
        """
        if bbox.left > page_left_margin:
            return bbox.left - page_left_margin
        return 0
    
    def detect_alignment(self, bbox: BoundingBox, 
                        page_width: float = 100.0,
                        tolerance: float = 5.0) -> str:
        """
        Detect text alignment from bbox.
        
        Args:
            bbox: Element's bounding box
            page_width: Page width as reference
            tolerance: Tolerance in percentage points
            
        Returns:
            'left', 'center', 'right', or 'justify'
        """
        left_margin = bbox.left
        right_margin = page_width - bbox.right
        
        # Check for center alignment
        if abs(left_margin - right_margin) < tolerance:
            return 'center'
        
        # Check for right alignment
        if left_margin > right_margin + tolerance:
            return 'right'
        
        # Check for justify (full width)
        if bbox.width > 90:
            return 'justify'
        
        # Default to left
        return 'left'
    
    def calculate_element_width(self, bbox: BoundingBox, 
                               page_width_inches: float = 8.5) -> float:
        """
        Convert bbox width to actual inches.
        
        Args:
            bbox: Element's bounding box
            page_width_inches: Page width in inches (typically 8.5 for letter)
            
        Returns:
            Element width in inches
        """
        return (bbox.width / 100.0) * page_width_inches


class SpacingEnhancer:
    """Enhanced element extraction with spacing information."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize enhancer.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = output_dir or Path('data/parsed/enhanced')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger('SpacingEnhancer')
        self.spacing_calc = SpacingCalculator()
    
    def enhance_extraction_results(self, extraction_json_path: Path, 
                                  output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Add spacing information to existing extraction results.
        
        Args:
            extraction_json_path: Path to docling_extraction_results.json
            output_path: Output path for enhanced results
            
        Returns:
            Enhanced extraction data with spacing info
        """
        self.logger.info(f"Enhancing extraction from: {extraction_json_path}")
        
        # Load original extraction
        with open(extraction_json_path, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
        
        # Enhance content structure
        content_structure = extraction_data.get('content_structure', {})
        reading_order = content_structure.get('reading_order', [])
        
        # Group elements by page
        elements_by_page = self._group_by_page(reading_order)
        
        # Calculate spacing for each page
        for page_num in sorted(elements_by_page.keys()):
            page_elements = elements_by_page[page_num]
            self._calculate_page_spacing(page_elements)
        
        # Save enhanced results
        if output_path is None:
            output_path = extraction_json_path.parent / 'extraction_with_spacing.json'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Saved enhanced extraction to: {output_path}")
        
        return extraction_data
    
    def _group_by_page(self, elements: List[Dict]) -> Dict[int, List[Dict]]:
        """Group elements by page number."""
        by_page = {}
        for elem in elements:
            page = elem.get('page', 1)
            if page not in by_page:
                by_page[page] = []
            by_page[page].append(elem)
        return by_page
    
    def _calculate_page_spacing(self, page_elements: List[Dict]):
        """Calculate spacing for elements on a page."""
        prev_bbox = None
        
        for i, elem in enumerate(page_elements):
            # Extract or create bbox
            bbox = self._extract_bbox(elem)
            if bbox is None:
                continue
            
            # Calculate spacing
            spacing = ElementSpacing()
            
            if i > 0:
                spacing.margin_top = self.spacing_calc.calculate_spacing_between(prev_bbox, bbox)
            
            # Infer properties
            indentation = self.spacing_calc.infer_indentation(bbox)
            alignment = self.spacing_calc.detect_alignment(bbox)
            
            # Store in element
            elem['bbox'] = bbox.to_dict()
            elem['spacing'] = spacing.to_dict()
            elem['layout_info'] = {
                'indentation': indentation,
                'alignment': alignment,
                'element_width': self.spacing_calc.calculate_element_width(bbox),
            }
            
            prev_bbox = bbox
    
    def _extract_bbox(self, element: Dict) -> Optional[BoundingBox]:
        """Extract bbox from element (if available)."""
        # Check if already has bbox info
        if 'bbox' in element and isinstance(element['bbox'], dict):
            bbox_dict = element['bbox']
            try:
                return BoundingBox(
                    left=float(bbox_dict.get('left', 0)),
                    top=float(bbox_dict.get('top', 0)),
                    right=float(bbox_dict.get('right', 100)),
                    bottom=float(bbox_dict.get('bottom', 100))
                )
            except (ValueError, TypeError):
                pass
        
        # Fallback: estimate based on element type and reading order
        # For now, return None - bbox would need to be extracted during initial parsing
        return None


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python spacing_extractor.py <extraction_json_path> [output_path]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Enhance extraction
    enhancer = SpacingEnhancer()
    result = enhancer.enhance_extraction_results(input_path, output_path)
    
    print(f"✓ Enhancement complete")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path or (input_path.parent / 'extraction_with_spacing.json')}")


if __name__ == '__main__':
    main()
