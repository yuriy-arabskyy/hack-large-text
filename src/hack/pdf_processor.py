"""PDF processing pipeline for RAG system.

This module combines the notebook experiments into a reusable pipeline:
1. Parse PDF with PyMuPDF (extract text blocks with bounding boxes)
2. Classify blocks by font size into section hierarchy
3. Generate embeddings using sentence-transformers
4. Build FAISS index for semantic search
"""

import pymupdf
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional
from pathlib import Path


def extract_text_blocks(page):
    """Extract text blocks with bounding boxes from a page."""
    text_blocks = []
    blocks = page.get_text("dict")["blocks"]

    for block_idx, block in enumerate(blocks):
        if block["type"] == 0:  # text block
            bbox = block["bbox"]
            text = ""

            # Collect font info from first span
            font_size = None
            font_name = None
            is_bold = False

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if font_size is None:
                        font_size = span.get("size", 10.0)
                        font_name = span.get("font", "Unknown")
                        is_bold = "Bold" in font_name
                    text += span["text"]
                text += "\n"

            text_blocks.append({
                "block_id": block_idx,
                "bbox": bbox,
                "text": text.strip(),
                "font_size": font_size or 10.0,
                "font_name": font_name or "Unknown",
                "is_bold": is_bold
            })

    return text_blocks


def classify_block_type(block: Dict, font_sizes: List[float]) -> str:
    """
    Classify block type based on font size and content.

    Returns: 'h1', 'h2', 'h3', 'body', or 'skip'
    """
    font_size = block["font_size"]
    text = block["text"]
    char_count = len(text)

    # Skip very short blocks or boilerplate
    if char_count < 3:
        return "skip"

    # Skip copyright/license text patterns
    skip_patterns = [
        "project gutenberg",
        "copyright",
        "license",
        "www.gutenberg.org",
        "ebook",
    ]
    text_lower = text.lower()
    if any(pattern in text_lower for pattern in skip_patterns) and char_count < 600:
        return "skip"

    # Compute font size percentiles for heading detection
    if len(font_sizes) > 0:
        p75 = np.percentile(font_sizes, 75)
        p90 = np.percentile(font_sizes, 90)
        p95 = np.percentile(font_sizes, 95)

        # Heading classification
        if font_size >= p95:
            return "h1"
        elif font_size >= p90:
            return "h2"
        elif font_size >= p75 and char_count < 100:
            return "h3"

    # Default to body text
    return "body"


def build_section_hierarchy(blocks: List[Dict]) -> List[Dict]:
    """
    Build section hierarchy and add section paths to blocks.

    Tracks h1/h2/h3 headings and assigns section_path to each block.
    """
    section_stack = []  # Stack of (level, title) tuples
    enhanced_blocks = []

    for block in blocks:
        block_type = block["type"]

        # Update section stack based on heading level
        if block_type == "h1":
            section_stack = [(1, block["text"])]
        elif block_type == "h2":
            # Pop h2 and h3 from stack, keep h1
            section_stack = [s for s in section_stack if s[0] < 2]
            section_stack.append((2, block["text"]))
        elif block_type == "h3":
            # Pop h3 from stack, keep h1 and h2
            section_stack = [s for s in section_stack if s[0] < 3]
            section_stack.append((3, block["text"]))

        # Build section path from stack
        section_path = " > ".join(s[1] for s in section_stack) if section_stack else None

        enhanced_block = block.copy()
        enhanced_block["section_path"] = section_path
        enhanced_blocks.append(enhanced_block)

    return enhanced_blocks


def parse_pdf(pdf_path: Path, doc_id: str = "uploaded_pdf") -> Dict:
    """
    Parse PDF and extract structured blocks with metadata.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier

    Returns:
        Workspace dictionary with blocks
    """
    doc = pymupdf.open(pdf_path)

    # First pass: extract all text blocks from all pages
    all_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_blocks = extract_text_blocks(page)

        for block in page_blocks:
            block["page_num"] = page_num
            block["char_count"] = len(block["text"])
            all_blocks.append(block)

    # Collect font sizes for classification
    font_sizes = [b["font_size"] for b in all_blocks]

    # Second pass: classify block types
    for block in all_blocks:
        block["type"] = classify_block_type(block, font_sizes)

    # Third pass: build section hierarchy
    all_blocks = build_section_hierarchy(all_blocks)

    # Re-index blocks with consistent IDs
    for idx, block in enumerate(all_blocks):
        block["block_idx"] = idx

    workspace = {
        "doc_id": doc_id,
        "num_pages": len(doc),
        "blocks": all_blocks
    }

    doc.close()
    return workspace


def generate_embeddings(workspace: Dict, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32) -> Dict:
    """
    Generate embeddings for all blocks in workspace.

    Args:
        workspace: Workspace dictionary from parse_pdf
        model_name: Sentence-transformers model name
        batch_size: Batch size for encoding

    Returns:
        Enhanced workspace with embeddings added to each block
    """
    model = SentenceTransformer(model_name)
    blocks = workspace["blocks"]

    # Collect texts to embed
    texts_to_embed = []
    text_indices = []

    for i, block in enumerate(blocks):
        if not block.get('text') or len(block['text'].strip()) < 3:
            text_indices.append(None)
        else:
            text_indices.append(len(texts_to_embed))
            texts_to_embed.append(block['text'].replace("\n", " "))

    # Generate embeddings
    if len(texts_to_embed) > 0:
        embeddings = model.encode(
            texts_to_embed,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
    else:
        embeddings = np.array([])

    # Add embeddings to blocks
    enhanced_blocks = []
    for i, block in enumerate(blocks):
        enhanced_block = block.copy()

        if text_indices[i] is None:
            enhanced_block['embedding'] = None
        else:
            enhanced_block['embedding'] = embeddings[text_indices[i]].tolist()

        enhanced_blocks.append(enhanced_block)

    workspace["blocks"] = enhanced_blocks
    return workspace


def build_faiss_index(workspace: Dict) -> tuple[faiss.Index, List[Dict]]:
    """
    Build FAISS index from workspace embeddings.

    Args:
        workspace: Workspace with embeddings

    Returns:
        Tuple of (FAISS index, list of valid blocks without embeddings)
    """
    # Extract embeddings and valid blocks
    embeddings = []
    valid_blocks = []

    for block in workspace["blocks"]:
        if block.get('embedding') is not None:
            embeddings.append(block['embedding'])
            # Store block without embedding to save memory
            block_copy = {k: v for k, v in block.items() if k != 'embedding'}
            valid_blocks.append(block_copy)

    if len(embeddings) == 0:
        raise ValueError("No valid embeddings found in workspace")

    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype='float32')
    dimension = embeddings_array.shape[1]

    # Create FAISS index with L2 distance
    index = faiss.IndexFlatL2(dimension)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)

    # Add vectors to index
    index.add(embeddings_array)

    return index, valid_blocks


def process_pdf(
    pdf_path: Path,
    doc_id: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2"
) -> tuple[faiss.Index, List[Dict], Dict]:
    """
    Complete PDF processing pipeline.

    Args:
        pdf_path: Path to PDF file
        doc_id: Document identifier (defaults to filename)
        model_name: Sentence-transformers model name

    Returns:
        Tuple of (FAISS index, valid blocks, workspace metadata)
    """
    if doc_id is None:
        doc_id = pdf_path.stem

    # Parse PDF
    workspace = parse_pdf(pdf_path, doc_id=doc_id)

    # Generate embeddings
    workspace = generate_embeddings(workspace, model_name=model_name)

    # Build FAISS index
    index, valid_blocks = build_faiss_index(workspace)

    # Metadata
    metadata = {
        "doc_id": workspace["doc_id"],
        "num_pages": workspace["num_pages"],
        "num_blocks": len(valid_blocks)
    }

    return index, valid_blocks, metadata
