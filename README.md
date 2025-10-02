# hack-large-text

A RAG (Retrieval-Augmented Generation) agent for querying large PDF documents with precise citations. Built with DSPy, FAISS, and Streamlit.

## Features

- 📄 **PDF Document Processing**: Parse and index large PDFs (100-300 pages)
- 🔍 **Semantic Search**: FAISS-based vector search for relevant content retrieval
- 🤖 **AI-Powered Q&A**: DSPy orchestration with OpenAI GPT-4o for intelligent question answering
- 📚 **Citation Tracking**: Answers include page numbers and section references
- 💬 **Chat Interface**: Interactive Streamlit UI for conversational document exploration

## Prerequisites

- Python ≥3.11
- Poetry for dependency management
- OpenAI API key

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hack-large-text
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

3. **Set up environment variables**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Running the Chat Application

Launch the Streamlit chat interface:

```bash
poetry run streamlit run src/hack/chat_app.py
```

The app will:
- Initialize the RAG agent with pre-indexed document data
- Load the FAISS vector index for semantic search
- Start a web interface at `http://localhost:8501`

## How It Works

1. **Query Understanding**: User questions are expanded into search terms and retrieval plans
2. **Evidence Retrieval**: FAISS searches the document index for relevant text blocks
3. **Evidence Selection**: The most relevant snippets are selected from candidates
4. **Answer Synthesis**: A comprehensive answer is generated with inline citations

## Project Structure

```
src/hack/
├── chat_app.py          # Streamlit chat interface
├── rag_agent.py         # RAG agent implementation
├── retriever.py         # FAISS retriever
├── models/
│   └── rag_models.py    # DSPy signature models
experiments/
├── chess_pdf.faiss      # FAISS vector index
└── workspace_with_embeddings.json  # Document metadata
```

## Development

**Run tests:**
```bash
poetry run pytest
```

**Lint/format code:**
```bash
poetry run ruff check .
poetry run ruff format .
```

**Run demo script (command-line):**
```bash
poetry run python src/hack/demo_rag.py
```

## Demo Dataset

The current demo uses "Chess Fundamentals" by José Capablanca as the indexed document. You can ask questions like:
- "What are the key principles for opening moves in chess?"
- "How should I approach the endgame?"
- "What is the importance of pawn structure?"

## License

See project license file.