from pathlib import Path
from docling.document_converter import DocumentConverter
from docling_core.types.doc import TextItem, PictureItem, DocItemLabel

pdf_path = Path("data/raw/FINTBX/pdf/split_pdfs/fintbx_part_001.pdf")
print("Converting:", pdf_path)
conv = DocumentConverter()
res = conv.convert(pdf_path)
doc = res.document
print("Document pages type:", type(doc.pages), "len:", len(doc.pages))

# Inspect first few formula TextItems
formula_count = 0
print("\n--- Formula elements (first 8) ---")
for element, level in doc.iterate_items():
    if isinstance(element, TextItem) and getattr(element, 'label', None) == DocItemLabel.FORMULA:
        print(f"Formula #{formula_count}: page prov={getattr(element, 'prov', None)}")
        print(" text repr:", repr(getattr(element, 'text', None)))
        print(" orig repr:", repr(getattr(element, 'orig', None))[:200])
        print(" available attrs:", [a for a in dir(element) if not a.startswith('_')][:40])
        formula_count += 1
        if formula_count >= 8:
            break

print(f"Total formulas found by iterate_items(): {formula_count}")

# Inspect picture items
print("\n--- Picture items (first 6) ---")
print("Total pictures (doc.pictures):", len(doc.pictures) if hasattr(doc, 'pictures') else 'N/A')
for i, pic in enumerate(doc.pictures[:6]):
    print(f"Picture #{i}: type={type(pic).__name__}, prov={getattr(pic, 'prov', None)}")
    try:
        img = pic.get_image(doc)
        print(" get_image returned type:", type(img))
        save_path = Path(f"tmp_picture_{i}.png")
        img.save(save_path)
        print(" saved to", save_path)
    except Exception as e:
        print(" get_image failed:", e)

print('\nDone')
