"""
examples/test_chunker_offline.py

Smoke test script to verify Phase 3 (Corpus Loading & Multi-Representation Chunking).
Loads the preprocessed JSON files from Phase 2 (ingestion/test_ingest_output) and runs LegalChunker.
Verifies that `title_page`, `section`, and `page_anchor` chunks are cleanly created with physical page numbers attached.
"""

import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.loaders import IngestedCorpusLoader
from retrieval.chunkers import LegalChunker, RecursiveChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_chunker_offline")


def main() -> None:
    logger.info("Initializing smoke test for Phase 3 (Loader & Chunker)...")

    # Point loader to where our Phase 2 smoke test output lives
    ingest_dir = Path("ingestion/test_ingest_output")
    if not ingest_dir.exists():
        logger.error("Ingest folder %s does not exist. Run Phase 2 test first.", ingest_dir)
        sys.exit(1)

    loader = IngestedCorpusLoader(ingest_dir=ingest_dir)
    loaded_docs = loader.load_all_documents()

    if not loaded_docs:
        logger.error("No documents loaded from %s.", ingest_dir)
        sys.exit(1)

    doc = loaded_docs[0]
    logger.info("Loaded document: '%s' with %d blocks across %d pages.",
                doc.doc_id, len(doc.blocks), doc.total_pages)

    # Initialize Chunker
    chunker = LegalChunker(max_section_chars=4000)
    chunks = chunker.chunk_document(doc)

    # Test Fallback Recursive Chunker (Sub-Step 3.4)
    rec_chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)
    rec_chunks = rec_chunker.chunk_document(doc)

    title_chunks = [c for c in chunks if c.representation == "title_page"]
    sec_chunks = [c for c in chunks if c.representation == "section"]
    page_chunks = [c for c in chunks if c.representation == "page_anchor"]

    logger.info("--- Phase 3 Chunker Results ---")
    logger.info("Total LegalChunks Created: %d", len(chunks))
    logger.info(" -> Title Page Chunks : %d (Pages cited: %s)", len(title_chunks), [c.pages for c in title_chunks])
    logger.info(" -> Section Chunks    : %d (Sample pages cited: %s)", len(sec_chunks), [c.pages for c in sec_chunks[:3]])
    logger.info(" -> Page Anchor Chunks: %d (Pages cited: %s)", len(page_chunks), [c.pages for c in page_chunks[:5]])
    logger.info(" -> Fallback Recursive Chunks (Sub-Step 3.4): %d (Sample pages: %s)", len(rec_chunks), [c.pages for c in rec_chunks[:3]])

    assert len(title_chunks) >= 1, "Must have at least 1 title_page chunk"
    assert len(sec_chunks) >= 1, "Must have at least 1 section chunk"
    assert len(page_chunks) == doc.total_pages, f"Must have exactly {doc.total_pages} page_anchor chunks"
    assert len(rec_chunks) >= 1, "Must generate fallback recursive chunks"

    print("\n" + "=" * 70)
    print("PHASE 3 SMOKE TEST PASSED SUCCESSFULLY across all 4 Sub-Steps!")
    print(f"Sample Section Chunk Text:\n{sec_chunks[0].text[:300]}...")
    print(f"\nSample Page Anchor Text (Page {page_chunks[0].pages[0]}):\n{page_chunks[0].text[:300]}...")
    print(f"\nSample Fallback Recursive Text (Page {rec_chunks[0].pages[0]}):\n{rec_chunks[0].text[:300]}...")
    print("=" * 70)


if __name__ == "__main__":
    main()
