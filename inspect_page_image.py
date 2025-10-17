from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import PictureItem

pdf_path = Path("data/raw/FINTBX/pdf/split_pdfs/fintbx_part_001.pdf")
print("Converting:", pdf_path)
conv = DocumentConverter()
res = conv.convert(str(pdf_path))
doc = res.document

print(f"Total pictures: {len(doc.pictures)}")
pic = doc.pictures[0]

print(f"\nFirst picture:")
print(f"  prov: {pic.prov}")
if pic.prov:
    page_no = pic.prov[0].page_no
    bbox = pic.prov[0].bbox
    print(f"  page_no: {page_no}")
    print(f"  bbox: {bbox}")
    
    # Try to get page and crop image
    print(f"\nTrying to extract image from page...")
    if hasattr(doc, 'pages'):
        print(f"doc.pages type: {type(doc.pages)}")
        # pages is a dict with page_no as keys
        page = doc.pages.get(page_no)
        print(f"page {page_no}: {type(page)}")
        
        if page and hasattr(page, 'image'):
            print(f"page.image: {type(page.image)}")
            if page.image:
                # Try to crop the image using bbox
                print(f"Attempting to crop image using bbox")
                print(f"page.image type: {type(page.image)}")
                # Check if it's a PIL image or has a pil_image attribute
                if hasattr(page.image, 'pil_image'):
                    pil_img = page.image.pil_image
                    print(f"Got PIL image: {pil_img.size}")
                    # Crop using bbox - need to convert bbox to pixel coordinates
                    # For now just save full page
                    pil_img.save("test_page_image.png")
                    print("Saved full page image as test_page_image.png")
                elif hasattr(page.image, 'save'):
                    page.image.save("test_page_image.png")
                    print("Saved page.image as test_page_image.png")
        
        # Alternative: check if there's a get_image method on page
        if hasattr(page, 'get_image'):
            print(f"page has get_image() method")
            try:
                img = page.get_image()
                print(f"page.get_image() returned: {type(img)}")
            except Exception as e:
                print(f"page.get_image() error: {e}")

print('\nDone')
