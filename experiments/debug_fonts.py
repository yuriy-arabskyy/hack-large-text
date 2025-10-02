"""Debug script to analyze PDF font properties and heading detection."""

import pymupdf
from pathlib import Path
from collections import Counter
import json

def analyze_fonts(pdf_path):
    """Deep dive into font properties across the PDF."""
    doc = pymupdf.open(pdf_path)

    all_font_info = []

    print(f"Analyzing {len(doc)} pages...\n")

    # Sample first 10 pages for detailed analysis
    for page_num in range(min(10, len(doc))):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block_idx, block in enumerate(blocks):
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        all_font_info.append({
                            "page": page_num,
                            "text": span["text"],
                            "size": round(span["size"], 2),
                            "font": span["font"],
                            "flags": span.get("flags", 0),
                            "color": span.get("color", 0),
                        })

    print("=" * 80)
    print("FONT SIZE DISTRIBUTION")
    print("=" * 80)
    sizes = [f["size"] for f in all_font_info]
    size_counter = Counter(sizes)
    for size, count in size_counter.most_common():
        print(f"  {size}pt: {count} spans")

    print("\n" + "=" * 80)
    print("FONT NAME DISTRIBUTION")
    print("=" * 80)
    fonts = [f["font"] for f in all_font_info]
    font_counter = Counter(fonts)
    for font, count in font_counter.most_common():
        print(f"  {font}: {count} spans")

    print("\n" + "=" * 80)
    print("FONT FLAGS DISTRIBUTION (bold, italic, etc.)")
    print("=" * 80)
    flags = [f["flags"] for f in all_font_info]
    flag_counter = Counter(flags)
    for flag, count in flag_counter.most_common():
        # Decode flags
        is_bold = flag & 2**4
        is_italic = flag & 2**1
        print(f"  Flags {flag}: {count} spans (bold={bool(is_bold)}, italic={bool(is_italic)})")

    print("\n" + "=" * 80)
    print("SAMPLE TEXT BY FONT SIZE (first 5 examples)")
    print("=" * 80)
    for size in sorted(set(sizes), reverse=True):
        samples = [f for f in all_font_info if f["size"] == size][:5]
        print(f"\n{size}pt ({len([f for f in all_font_info if f['size'] == size])} total):")
        for sample in samples:
            text_preview = sample["text"][:60].strip()
            if text_preview:
                print(f"  Page {sample['page']}: {text_preview}")

    print("\n" + "=" * 80)
    print("SAMPLE TEXT BY FONT NAME")
    print("=" * 80)
    for font_name in font_counter.keys():
        samples = [f for f in all_font_info if f["font"] == font_name][:3]
        print(f"\n{font_name}:")
        for sample in samples:
            text_preview = sample["text"][:60].strip()
            if text_preview:
                print(f"  {sample['size']}pt: {text_preview}")

    # Look for potential headings (short text, different properties)
    print("\n" + "=" * 80)
    print("POTENTIAL HEADINGS (short text < 80 chars)")
    print("=" * 80)

    # Group by block and analyze
    doc = pymupdf.open(pdf_path)
    for page_num in range(min(5, len(doc))):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span["text"]

                text = text.strip()
                if text and len(text) < 80 and len(text) > 10:
                    # Get font info
                    first_span = block["lines"][0]["spans"][0]
                    print(f"\nPage {page_num}: '{text[:60]}'")
                    print(f"  Font: {first_span['font']}, Size: {first_span['size']}, Flags: {first_span.get('flags', 0)}")


def detect_headings_by_pattern(pdf_path):
    """Detect headings using text patterns instead of font properties."""
    doc = pymupdf.open(pdf_path)

    print("\n" + "=" * 80)
    print("PATTERN-BASED HEADING DETECTION")
    print("=" * 80)

    headings = []

    for page_num in range(min(20, len(doc))):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if block["type"] == 0:
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span["text"]

                text = text.strip()

                if not text or len(text) < 3:
                    continue

                # Heading heuristics for plain text PDFs
                is_all_caps = text.isupper() and len(text) > 5
                is_short = len(text) < 80
                has_chapter = text.startswith("CHAPTER")
                has_number_prefix = text[0].isdigit() and ". " in text[:5]
                is_underscored = text.startswith("_") and text.endswith("_")

                heading_level = None

                if has_chapter or (is_all_caps and is_short and not text.startswith("***")):
                    heading_level = "h1"
                elif has_number_prefix and is_short:
                    heading_level = "h2"
                elif is_underscored or (is_short and len(text) > 10 and text.count(" ") < 8):
                    # Short lines that aren't just a few words
                    if not any(skip in text.lower() for skip in ["illustration", "copyright", "printed"]):
                        heading_level = "h3"

                if heading_level:
                    headings.append({
                        "page": page_num,
                        "level": heading_level,
                        "text": text[:80]
                    })

    print(f"\nFound {len(headings)} potential headings:\n")
    for h in headings[:50]:
        indent = "  " * (int(h["level"][1]) - 1)
        print(f"{indent}[{h['level'].upper()}] Page {h['page']}: {h['text']}")

    if len(headings) > 50:
        print(f"\n... and {len(headings) - 50} more")

    return headings


if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent / "resources" / "chess.pdf"
    analyze_fonts(pdf_path)
    detect_headings_by_pattern(pdf_path)
