"""FAISS-based retriever for RAG with sentence-transformers embeddings."""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from pathlib import Path


class FaissRetriever:
    """Retriever that uses FAISS index for semantic search over document blocks."""

    def __init__(
        self,
        faiss_index_path: Optional[str] = None,
        workspace_json_path: Optional[str] = None,
        faiss_index: Optional[faiss.Index] = None,
        blocks: Optional[List[Dict]] = None,
        model_name: str = "all-MiniLM-L6-v2",
        k: int = 10
    ):
        """
        Initialize FAISS retriever.

        Two modes:
        1. File-based: Provide faiss_index_path and workspace_json_path
        2. In-memory: Provide faiss_index and blocks directly

        Args:
            faiss_index_path: Path to the FAISS index file (file-based mode)
            workspace_json_path: Path to workspace JSON with blocks (file-based mode)
            faiss_index: FAISS index object (in-memory mode)
            blocks: List of blocks without embeddings (in-memory mode)
            model_name: Name of sentence-transformers model (must match embedding model)
            k: Default number of results to return
        """
        self.k = k

        # Determine mode and load data
        if faiss_index is not None and blocks is not None:
            # In-memory mode
            self.index = faiss_index
            self.blocks = blocks
        elif faiss_index_path is not None and workspace_json_path is not None:
            # File-based mode
            index_path = Path(faiss_index_path)
            if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")
            self.index = faiss.read_index(str(index_path))

            # Load workspace data
            workspace_path = Path(workspace_json_path)
            if not workspace_path.exists():
                raise FileNotFoundError(f"Workspace JSON not found at {workspace_json_path}")

            with open(workspace_path, 'r') as f:
                self.workspace_data = json.load(f)

            # Extract blocks with valid embeddings
            self.blocks = []
            for block in self.workspace_data['blocks']:
                if block.get('embedding') is not None:
                    # Store block without embedding to save memory
                    block_copy = {k: v for k, v in block.items() if k != 'embedding'}
                    self.blocks.append(block_copy)
        else:
            raise ValueError(
                "Must provide either (faiss_index + blocks) for in-memory mode "
                "or (faiss_index_path + workspace_json_path) for file-based mode"
            )

        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.blocks)} blocks from workspace")

    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query text into embedding vector."""
        embedding = self.model.encode([query], convert_to_numpy=True)
        embedding = embedding.astype('float32')
        # Normalize for cosine similarity (matching index creation)
        faiss.normalize_L2(embedding)
        return embedding

    def _search(self, query: str, k: Optional[int] = None, block_type: Optional[str] = None) -> List[Dict]:
        """
        Search FAISS index and return matching blocks.

        Args:
            query: Search query text
            k: Number of results to return (uses default if None)
            block_type: Filter by block type (e.g., 'body', 'heading', 'skip')

        Returns:
            List of blocks with metadata
        """
        if k is None:
            k = self.k

        # Encode query
        query_embedding = self._encode_query(query)

        # Search FAISS index
        # Request more results if we need to filter by type
        search_k = k * 3 if block_type else k
        distances, indices = self.index.search(query_embedding, search_k)

        # Retrieve matching blocks
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(self.blocks):
                continue

            block = self.blocks[idx].copy()

            # Filter by type if specified
            if block_type and block.get('type') != block_type:
                continue

            # Add similarity score
            # For normalized L2 distance: distance is in [0, 2], convert to similarity [0, 1]
            # Lower distance = higher similarity
            similarity = max(0.0, 1.0 - (distance / 2.0))
            block['similarity'] = float(similarity)

            # Format as expected by WorkspaceAgent
            result = {
                'unit_id': f"block_{block['page_num']}_{block['block_idx']}",
                'content': block['text'],
                'page': block['page_num'],
                'section_path': block.get('section_path', ''),
                'bbox': block.get('bbox', []),
                'type': block.get('type', 'unknown'),
                'similarity': block['similarity']
            }

            results.append(result)

            if len(results) >= k:
                break

        return results

    def search_text(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Search for text blocks matching the query.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of text blocks with metadata
        """
        # Don't filter by type - return all text content
        return self._search(query, k=k, block_type=None)

    def search_tables(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Search for table blocks matching the query.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of table blocks with metadata
        """
        # Currently no tables in the chess PDF, but filter by type
        return self._search(query, k=k, block_type='table')

    def search_images(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Search for image/figure blocks matching the query.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of image blocks with metadata
        """
        # Currently no images in the chess PDF, but filter by type
        return self._search(query, k=k, block_type='image')

    def search_all(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """
        Search all blocks regardless of type.

        Args:
            query: Search query text
            k: Number of results to return

        Returns:
            List of all matching blocks with metadata
        """
        return self._search(query, k=k, block_type=None)


class MockRetriever:
    """Mock retriever for testing purposes."""

    def search_text(self, query: str) -> List[Dict]:
        """Return mock text results."""
        return [
            {
                'unit_id': 'mock_1',
                'content': 'Mock text result',
                'page': 1,
                'section_path': '1.0',
                'bbox': [0, 0, 100, 100]
            }
        ]

    def search_tables(self, query: str) -> List[Dict]:
        """Return mock table results."""
        return []

    def search_images(self, query: str) -> List[Dict]:
        """Return mock image results."""
        return []
