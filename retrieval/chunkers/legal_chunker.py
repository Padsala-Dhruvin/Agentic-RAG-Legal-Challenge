"""
retrieval/chunkers/legal_chunker.py

Multi-Representation Legal Chunker (`title_page`, `section`, `page_anchor`)
optimizing both retrieval precision ($A$) and physical page grounding ($G$).
Implements BaseChunker interface.
"""

import logging
from typing import Any, Dict, List

from retrieval.chunkers.base import BaseChunker
from retrieval.chunkers.legal_chunk_types import LegalChunk
from retrieval.loaders.ingested_corpus_loader import LoadedDocument, TextBlock

logger = logging.getLogger(__name__)


class LegalChunker(BaseChunker):
    """Slices LoadedDocuments into 3 specialized representations (`title_page`, `section`, `page_anchor`)."""

    def __init__(self, max_section_chars: int = 4000) -> None:
        self.max_section_chars: int = max_section_chars

    def _chunk_title_page(self, doc: LoadedDocument) -> List[LegalChunk]:
        """Representation A: Document header / cover page summary."""
        meta_lines = [f"Document ID: {doc.doc_id}"]
        for k, v in doc.metadata.items():
            if v and k not in ("doc_id",):
                meta_lines.append(f"{k.replace('_', ' ').title()}: {v}")

        first_blocks = [b.text for b in doc.blocks if b.page_idx == 0][:3]
        summary_text = "\n".join(meta_lines) + "\n\n--- Title Page Excerpt ---\n" + "\n\n".join(first_blocks)

        return [LegalChunk(
            chunk_id=f"{doc.doc_id}_title_page",
            doc_id=doc.doc_id,
            representation="title_page",
            text=summary_text.strip(),
            pages=[1],
            metadata={"doc_type": doc.metadata.get("doc_type", "unknown")},
        )]

    def _chunk_sections(self, doc: LoadedDocument) -> List[LegalChunk]:
        """Representation B: Bounded heading-to-heading section windows."""
        chunks = []
        current_heading = "General / Preamble"
        current_blocks: List[TextBlock] = []
        section_counter = 0

        def _flush_window() -> None:
            nonlocal section_counter
            if not current_blocks:
                return
            text = "\n\n".join([f"[{current_heading}]"] + [b.text for b in current_blocks])
            pages = sorted(list({b.page_idx + 1 for b in current_blocks}))
            chunk_id = f"{doc.doc_id}_sec_{section_counter}"
            chunks.append(LegalChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                representation="section",
                text=text.strip(),
                pages=pages,
                metadata={"heading": current_heading, "doc_type": doc.metadata.get("doc_type")},
            ))
            section_counter += 1

        for block in doc.blocks:
            if block.level in (1, 2, 3) and len(block.text) < 200:
                # Flush existing window before starting a new heading
                _flush_window()
                current_heading = block.text.strip().replace("\n", " ")
                current_blocks = [block]
            else:
                current_blocks.append(block)
                # If section grows too large, split cleanly at boundary
                if sum(len(b.text) for b in current_blocks) >= self.max_section_chars:
                    _flush_window()
                    current_blocks = []

        _flush_window()
        return chunks

    def _chunk_page_anchors(self, doc: LoadedDocument) -> List[LegalChunk]:
        """Representation C: Exact physical page windows with heading breadcrumbs."""
        page_map: Dict[int, List[TextBlock]] = {}
        for block in doc.blocks:
            page_map.setdefault(block.page_idx, []).append(block)

        chunks = []
        active_heading = "General"
        for page_idx in sorted(page_map.keys()):
            blocks = page_map[page_idx]
            # Update breadcrumb if page has a header
            for b in blocks:
                if b.level in (1, 2, 3) and len(b.text) < 200:
                    active_heading = b.text.strip().replace("\n", " ")
                    break

            text = f"[Breadcrumb: {active_heading} | Physical Page {page_idx + 1}]\n\n" + "\n\n".join(b.text for b in blocks)
            chunks.append(LegalChunk(
                chunk_id=f"{doc.doc_id}_page_{page_idx + 1}",
                doc_id=doc.doc_id,
                representation="page_anchor",
                text=text.strip(),
                pages=[page_idx + 1],
                metadata={"breadcrumb": active_heading, "page_number": page_idx + 1},
            ))

        return chunks

    def chunk_document(self, doc: LoadedDocument) -> List[LegalChunk]:
        """Run all 3 representation methods on a LoadedDocument."""
        title_chunks = self._chunk_title_page(doc)
        section_chunks = self._chunk_sections(doc)
        page_chunks = self._chunk_page_anchors(doc)
        return title_chunks + section_chunks + page_chunks

    def chunk_all_documents(self, docs: List[LoadedDocument]) -> List[LegalChunk]:
        """Chunk a list of documents and log representation counts."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))

        counts = {r: sum(1 for c in all_chunks if c.representation == r) for r in ("title_page", "section", "page_anchor")}
        logger.info("Chunked %d documents into %d total chunks (%s)", len(docs), len(all_chunks), counts)
        return all_chunks
