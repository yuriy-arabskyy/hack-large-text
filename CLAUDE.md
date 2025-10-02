# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for building an agent-based document processing system to handle large PDFs (100-300 pages). The architecture focuses on creating persistent workspaces that parse, index, and enable querying of long documents with precise citations.

See `resources/plan.md` for the complete architectural vision covering workspace creation (PDF ingestion/indexing) and workspace usage (Q&A, summaries with citations).

## Development Commands

**Environment Setup:**
```bash
poetry install
```

**Run Tests:**
```bash
poetry run pytest
```

**Run Single Test:**
```bash
poetry run pytest tests/hack/test_main.py::test_agent_creation
```

**Lint/Format:**
```bash
poetry run ruff check .
poetry run ruff format .
```

**Run Main Script:**
```bash
poetry run python src/hack/main.py
```

## Architecture

- **Agent Framework**: Uses `openai-agents` library for LLM-powered agent orchestration
- **Project Structure**:
  - `src/hack/` - Main application code
  - `tests/` - Test suite with matching structure
  - `resources/` - Planning docs and specifications
- **Python Version**: Requires Python ≥3.12
- **Planned Components** (per plan.md):
  - PDF parsing & segmentation (text blocks, tables, figures with bounding boxes)
  - Database storage with full-text and semantic indexing
  - Evidence-based retrieval with precise citations (page, section, bbox)
  - Coverage tracking and incremental update capabilities
