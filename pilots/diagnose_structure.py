#!/usr/bin/env python3
"""
Diagnostic script to understand DoclingDocument v2.49.0 structure
"""

from pathlib import Path
from docling.document_converter import DocumentConverter

def main():
    # Convert just first 5 pages
    pdf_path = Path("data/raw/FINTBX/pdf/split_pdfs/fintbx_part_001.pdf")
    
    print(f"Converting: {pdf_path}")
    converter = DocumentConverter()
    result = converter.convert(pdf_path, page_range=(1, 5))
    doc = result.document
    
    print(f"\n✓ Document converted successfully")
    print(f"  Total pages: {len(doc.pages)}")
    print(f"  Document type: {type(doc).__name__}")
    
    print(f"\n📋 DOCUMENT-LEVEL ATTRIBUTES:")
    print(f"  has 'pages': {hasattr(doc, 'pages')}")
    print(f"  has 'tables': {hasattr(doc, 'tables')}")
    print(f"  has 'pictures': {hasattr(doc, 'pictures')}")
    print(f"  has 'blocks': {hasattr(doc, 'blocks')}")
    
    if hasattr(doc, 'tables'):
        print(f"  → tables count: {len(doc.tables)}")
    if hasattr(doc, 'pictures'):
        print(f"  → pictures count: {len(doc.pictures)}")
    
    # Check pages structure
    print(f"\n📄 PAGES STRUCTURE:")
    print(f"  type(doc.pages): {type(doc.pages).__name__}")
    print(f"  len(doc.pages): {len(doc.pages)}")
    
    # Check if it's dict or list
    if isinstance(doc.pages, dict):
        print(f"  Keys: {list(doc.pages.keys())[:5]}...")  # Show first 5 keys
        page_key = list(doc.pages.keys())[0]
        page = doc.pages[page_key]
    else:
        print(f"  It's a list/sequence, accessing [0]")
        page = doc.pages[0] if len(doc.pages) > 0 else None
    
    if page:
        print(f"\n� FIRST PAGE STRUCTURE:")
        print(f"  type: {type(page).__name__}")
        print(f"  has 'blocks': {hasattr(page, 'blocks')}")
        
        # List all attributes that might contain content
        attrs = [attr for attr in dir(page) if not attr.startswith('_')]
        print(f"  Attributes (first 15): {attrs[:15]}")
        
        if hasattr(page, 'blocks'):
            print(f"  blocks count: {len(page.blocks) if page.blocks else 0}")
            if page.blocks:
                # Show types of blocks
                block_types = {}
                for block in page.blocks:
                    btype = type(block).__name__
                    block_types[btype] = block_types.get(btype, 0) + 1
                print(f"  Block types: {block_types}")
    
    # Use iterate_items (the CORRECT API for v2.49.0)
    print(f"\n🔄 USING iterate_items() - THE CORRECT API:")
    from docling_core.types.doc import PictureItem, TableItem, TextItem
    
    element_types = {}
    total_count = 0
    
    for element, level in doc.iterate_items():
        elem_type = type(element).__name__
        element_types[elem_type] = element_types.get(elem_type, 0) + 1
        total_count += 1
    
    print(f"  Total elements: {total_count}")
    print(f"  Element types:")
    for elem_type, cnt in sorted(element_types.items(), key=lambda x: -x[1]):
        print(f"    - {elem_type}: {cnt}")
    
    # Show examples
    print(f"\n📊 ELEMENT EXAMPLES:")
    count = 0
    for element, level in doc.iterate_items():
        if isinstance(element, TextItem):
            if count < 3:
                print(f"  TextItem: {element.text[:60]}...")
                count += 1
                
    if doc.pictures:
        print(f"\n🖼️ PICTURES:")
        for i, pic in enumerate(doc.pictures[:3]):
            print(f"  Picture {i+1}: {type(pic).__name__}")
            if hasattr(pic, 'caption'):
                print(f"    Caption: {pic.caption}")
    
    if doc.tables:
        print(f"\n📑 TABLES:")
        for i, table in enumerate(doc.tables[:3]):
            print(f"  Table {i+1}: {type(table).__name__}")
            try:
                df = table.export_to_dataframe()
                print(f"    Shape: {df.shape}")
            except Exception as e:
                print(f"    Error exporting: {e}")

if __name__ == "__main__":
    main()
