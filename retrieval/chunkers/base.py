"""
retrieval/chunkers/base.py

Abstract base class for all chunkers in the pipeline.
"""

from abc import ABC, abstractmethod
from typing import List

from retrieval.chunkers.legal_chunk_types import LegalChunk
from retrieval.loaders.ingested_corpus_loader import LoadedDocument


class BaseChunker(ABC):
    """Abstract interface for legal chunking strategies."""

    @abstractmethod
    def chunk_document(self, doc: LoadedDocument) -> List[LegalChunk]:
        """Slice a LoadedDocument into a list of LegalChunks."""
        pass

    def chunk_all_documents(self, docs: List[LoadedDocument]) -> List[LegalChunk]:
        """Slice multiple documents into chunks."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
