"""
retrieval/chunkers/recursive_chunker.py

Fallback Recursive Character Splitter implementing BaseChunker.
Used when unstructured plain text or raw un-ingested files are encountered,
providing graceful degradation while tracking estimated physical page boundaries.
"""

import logging
from typing import List

from retrieval.chunkers.base import BaseChunker
from retrieval.chunkers.legal_chunk_types import LegalChunk
from retrieval.loaders.ingested_corpus_loader import LoadedDocument

logger = logging.getLogger(__name__)


class RecursiveChunker(BaseChunker):
    """Fallback recursive character chunker with overlap and page tracking."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " "]

    def _split_text(self, text: str) -> List[str]:
        """Split string recursively using hierarchy of natural separators."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        for sep in self.separators:
            if sep in text:
                parts = text.split(sep)
                chunks = []
                current_chunk = ""
                for part in parts:
                    candidate = f"{current_chunk}{sep}{part}" if current_chunk else part
                    if len(candidate) <= self.chunk_size:
                        current_chunk = candidate
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        # Handle huge individual parts by recursing or slicing
                        if len(part) > self.chunk_size and sep != " ":
                            sub_chunks = self._split_text(part)
                            chunks.extend(sub_chunks[:-1])
                            current_chunk = sub_chunks[-1] if sub_chunks else ""
                        else:
                            current_chunk = part
                if current_chunk:
                    chunks.append(current_chunk.strip())
                return [c for c in chunks if c]

        # If no separators worked, hard slice by characters
        return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

    def chunk_document(self, doc: LoadedDocument) -> List[LegalChunk]:
        """Slice a LoadedDocument or plain text into recursive chunks."""
        chunks = []
        counter = 0

        # If we have structured blocks from Phase 2, chunk block-by-block maintaining page_idx
        if doc.blocks:
            for block in doc.blocks:
                sub_texts = self._split_text(block.text)
                for st in sub_texts:
                    chunks.append(LegalChunk(
                        chunk_id=f"{doc.doc_id}_rec_{counter}",
                        doc_id=doc.doc_id,
                        representation="recursive",
                        text=st,
                        pages=[block.page_idx + 1],
                        metadata={"block_type": block.block_type, "fallback": True},
                    ))
                    counter += 1
        else:
            # Entirely raw string document without blocks
            raw_text = str(doc.metadata.get("raw_text", ""))
            sub_texts = self._split_text(raw_text)
            for idx, st in enumerate(sub_texts):
                # Estimate 1-based page index (assuming ~3000 chars per page)
                est_page = max(1, (idx * self.chunk_size) // 3000 + 1)
                chunks.append(LegalChunk(
                    chunk_id=f"{doc.doc_id}_rec_{counter}",
                    doc_id=doc.doc_id,
                    representation="recursive",
                    text=st,
                    pages=[est_page],
                    metadata={"fallback": True, "estimated_page": est_page},
                ))
                counter += 1

        return chunks
