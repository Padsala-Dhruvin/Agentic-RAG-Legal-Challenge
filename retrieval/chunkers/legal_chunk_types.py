"""
retrieval/chunkers/legal_chunk_types.py

Defines the core LegalChunk data class used across all chunking strategies
and downstream indexing engines.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LegalChunk:
    """Represents a finalized chunk ready for embedding, BM25 indexing, and retrieval.

    Attributes:
        chunk_id: Unique identifier for the chunk (e.g. 'doc123_sec_0').
        doc_id: The document identifier (`doc_id`).
        representation: The chunk representation type ('title_page', 'section', 'page_anchor', 'recursive').
        text: The actual text content of the chunk.
        pages: List of 1-based physical page numbers where this chunk appears (required for Grounding score $G$).
        metadata: Additional metadata (e.g., breadcrumb heading, section path, doc_type).
    """
    chunk_id: str
    doc_id: str
    representation: str
    text: str
    pages: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)
