from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import PictureItem
import json

pdf_path = Path("data/raw/FINTBX/pdf/split_pdfs/fintbx_part_001.pdf")
print("Converting:", pdf_path)
conv = DocumentConverter()
res = conv.convert(str(pdf_path))
doc = res.document

print(f"\nTotal pictures: {len(doc.pictures)}")
pic = doc.pictures[0]
print(f"\nFirst picture analysis:")
print(f"Type: {type(pic)}")
print(f"Dir (non-private): {[a for a in dir(pic) if not a.startswith('_')]}")

# Try all possible ways to get image
print("\n=== Attempting to extract image ===")

# Check various attributes
attrs_to_check = ['image', 'image_ref', 'image_data', 'get_image', 'picture', 'ref']
for attr in attrs_to_check:
    if hasattr(pic, attr):
        val = getattr(pic, attr)
        print(f"\n{attr}: {type(val).__name__}")
        if callable(val):
            try:
                result = val()
                print(f"  {attr}() returned: {type(result)}")
            except TypeError:
                try:
                    result = val(doc)
                    print(f"  {attr}(doc) returned: {type(result)}")
                except Exception as e:
                    print(f"  {attr}(doc) error: {e}")
            except Exception as e:
                print(f"  {attr}() error: {e}")
        else:
            print(f"  value: {val}")

# Check if it's a ref (reference to document image)
if hasattr(pic, 'ref'):
    print(f"\nPicture has ref: {pic.ref}")
    print(f"Is ref in document images? {pic.ref in doc.embedded_images if hasattr(doc, 'embedded_images') else 'N/A'}")

# Try to access via document
print(f"\nDocument attributes: {[a for a in dir(doc) if 'image' in a.lower()]}")

# Check if document has embedded_images or pictures_with_data
if hasattr(doc, 'embedded_images'):
    print(f"Document has embedded_images: {type(doc.embedded_images)}, len={len(doc.embedded_images)}")

# Try the export_to_dict approach
print("\n=== Checking exported data ===")
try:
    pic_dict = pic.model_dump() if hasattr(pic, 'model_dump') else pic.dict()
    keys = list(pic_dict.keys())
    print(f"Picture dict keys: {keys}")
    for key in ['image', 'image_ref', 'ref', 'data']:
        if key in pic_dict:
            val = pic_dict[key]
            print(f"  {key}: {type(val).__name__} = {str(val)[:100]}")
except Exception as e:
    print(f"Export error: {e}")

print('\nDone')
