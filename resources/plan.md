# Document Workspace Workflow

This document describes the end-to-end workflow for handling long, mixed-content PDFs (100–300 pages) in Track B.  
The process is split into two phases:

1. **Workspace Creation** (ingestion & indexing)  
2. **Workspace Usage** (Q&A, summaries, briefs, with citations)

---

## 1. Workspace Creation

**Input:** PDF (manual, policy, report, etc.)  
**Output:** A persistent workspace with structured units (text, tables, figures), stored in a database and searchable.

### Steps

1. **Parse & Segment**
   - Use `pymupdf`/`pdfplumber` to split into **pages**.
   - Extract **text blocks** (paragraphs + bounding boxes).
   - Extract **tables** (structured if possible; fallback = image crops).
   - Extract **figures/images** with bounding boxes.
   - OCR fallback for scanned pages.

2. **Annotate**
   - Detect headings → build section hierarchy / TOC.
   - Attach `section_path` to each block, table, figure.

3. **Assign IDs**
   - `doc_id` (e.g., UUID or filename-based).  
   - `page_id` = `(doc_id, page_no)`.  
   - `unit_id` = `(doc_id, page_no, unit_index)` for each block/table/figure.  

4. **Persist**
   - Store structured data in **DB**:
     - `pages`, `blocks`, `tables`, `figures`
   - Store binary assets (image crops, thumbnails) in FS/S3.
   - Create indices:
     - **Full-text** (BM25 / SQLite FTS5).
     - **Optional semantic index** (Qdrant, per modality).

5. **Precompute Metadata**
   - Generate image captions.
   - Generate table surrogates (headers, sample rows).
   - Run lightweight NER for entities/dates.
   - Store alongside units.

**Result:**  
A “workspace” keyed by `doc_id` where each evidence unit (block/table/figure) has:
- Text/content
- Section path
- Page number
- Bounding box
- Stored in DB + searchable

---

## 2. Workspace Usage

Once a workspace is created, agents can serve queries or generate summaries.  
**Principle:** Agent never reads the whole PDF—only retrieves *small units* with anchors.

### Q&A Flow

1. **User asks a question**  
   Example: *“What are the warranty exclusions in the manual?”*

2. **Agent planning**
   - Rewrite query into search terms.
   - Call `search_text`, `search_tables`, `search_images`.

3. **Retrieval**
   - BM25 search over blocks.
   - Optional vector search over tables/images.
   - Return candidate units (with IDs + `(doc_id, page, bbox, section_path)`).

4. **Evidence assembly**
   - Fetch exact snippets (`get_text(unit_id)`).
   - If figure/table: fetch crop or structured JSON.

5. **Answer synthesis**
   - Agent composes natural-language answer.  
   - Every sentence linked to one or more citations.  
   - Inline tags `[p.47 §2.3]`.  
   - JSON sidecar with exact anchors:
     ```json
     [
       {"unit_id": "block_123", "doc_id": "manual_v1", "page": 47, "bbox": [100,200,400,250]}
     ]
     ```

6. **Return result**
   - Human-readable answer + clickable citations.  
   - Provenance log stored.

---

### Executive Brief Flow

1. Agent calls `outline(doc_id)` for section hierarchy.
2. Iterates per section:
   - Summarizes top blocks/tables/figures.
   - Adds citations for each claim.
3. Reduces into a **one-page brief**.
4. Outputs:
   - `executive_brief.md` (inline `[p.X §Y]` citations).
   - `executive_brief.citations.json` (anchors for UI).

---

### Coverage Report

While generating summaries or Q&A:
- Track which units are cited.  
- Report:
  - % of sections covered
  - # tables and figures cited
  - Pages without coverage
  - Heatmap (optional)

---

## Example Timeline

- **Day 0:** Ingest PDF → workspace created. (~2,000 text blocks, 120 tables, 90 figures)  
- **Day 1:** Ask Q&A → Answer with citations `[p.46 §Warranty Terms]`.  
- **Day 2:** Generate executive brief → one-page summary with clickable anchors.  

---

## Deliverables

- **Workspace creation pipeline**  
- **Q&A with citations**  
- **Executive brief with JSON anchors**  
- **Coverage report**  

---

## Key Benefits

- **Accuracy:** Every claim backed by page/region citation.  
- **Efficiency:** Retrieve only minimal units; tokens + latency optimized.  
- **Reliability:** Deterministic DB storage; citations reproducible.  
- **Reusability:** Persisted workspace enables fast, cheap follow-up queries.  
- **Auditability:** Provenance logs for all answers.