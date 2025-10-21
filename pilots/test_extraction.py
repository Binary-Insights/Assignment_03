#!/usr/bin/env python3
"""
Quick test script to verify the fixed extraction methods work
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from parse.docling_extractor_v2 import DoclingExtractor

def main():
    # Test parameters
    pdf_path = Path("data/raw/FINTBX/pdf/split_pdfs/fintbx_part_001.pdf")
    pdf_name = "fintbx_part_001.pdf"
    
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        return
    
    print(f"Testing extraction with: {pdf_path}")
    print("=" * 60)
    
    # Initialize extractor
    extractor = DoclingExtractor(output_dir="data/parsed/FINTBX", company_ticker="FINTBX")
    
    # Run extraction
    results = extractor.extract_from_pdf(str(pdf_path))
    
    # Print results
    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS:")
    print("=" * 60)
    
    content_struct = results.get('content_structure', {})
    
    print(f"\nReading Order Elements: {len(content_struct.get('reading_order', []))}")
    print(f"Tables: {content_struct.get('tables', {}).get('count', 0)}")
    print(f"Figures: {content_struct.get('figures', {}).get('count', 0)}")
    print(f"Formulas: {content_struct.get('formulas', {}).get('count', 0)}")
    
    # Show first few reading order elements
    reading_order = content_struct.get('reading_order', [])
    if reading_order:
        print(f"\nFirst 5 reading order elements:")
        for i, elem in enumerate(reading_order[:5]):
            print(f"  {i+1}. Page {elem.get('page')}, Type: {elem.get('type')}, Content: {elem.get('content', '')[:50]}...")
    
    # Check for errors
    if content_struct.get('figures', {}).get('error'):
        print(f"\nFigures Error: {content_struct['figures']['error']}")
    if content_struct.get('formulas', {}).get('error'):
        print(f"Formulas Error: {content_struct['formulas']['error']}")
    
    print(f"\nProcessing time: {results.get('processing_time_seconds', 0):.2f} seconds")
    print("=" * 60)

if __name__ == "__main__":
    main()
