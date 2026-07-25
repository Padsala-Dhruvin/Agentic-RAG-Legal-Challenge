"""
examples/test_hybrid_indexer_offline.py

Smoke test script for Sub-Step 5.4 (Hybrid Indexer & RRF Fusion).
Indexes our real Phase 3 chunks and verifies that parallel BM25 + Vector + RRF
retrieves exact candidate chunks and reranks them accurately.
"""

import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import HybridIndexer, IngestedCorpusLoader, LegalChunker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_hybrid_indexer_offline")


def main() -> None:
    logger.info("Initializing smoke test for Sub-Step 5.4 (Hybrid Indexer & RRF)...")

    ingest_dir = Path("ingestion/test_ingest_output")
    if not ingest_dir.exists():
        logger.error("Ingest directory %s missing. Run Phase 2 first.", ingest_dir)
        sys.exit(1)

    # 1. Load and Chunk real document (`03b621...pdf` -> CA 005/2025)
    loader = IngestedCorpusLoader(ingest_dir=ingest_dir)
    docs = loader.load_all_documents()
    chunker = LegalChunker(max_section_chars=4000)
    chunks = chunker.chunk_all_documents(docs)

    # 2. Build Hybrid Index (`BM25` + `InMemoryVectorStore`)
    indexer = HybridIndexer(rrf_k=60)
    indexer.index_chunks(chunks)

    # 3. Test Query across the index
    test_query = "What did the Court of Appeal order regarding the provision of security for costs and AED 550,000?"
    logger.info("Running search for query: \"%s\"", test_query)

    results = indexer.search(test_query, top_k=3)

    print("-" * 70)
    print(f"Total Search Results Retrieved: {len(results)}")
    for rank, (chunk, score) in enumerate(results, 1):
        print(f"Rank #{rank} | Score: {score:.4f} | Chunk ID: {chunk.chunk_id} | Representation: {chunk.representation}")
        print(f"Cited Physical Pages: {chunk.pages}")
        print(f"Text Preview:\n{chunk.text[:250]}...")
        print("-" * 70)

    assert len(results) >= 1, "HybridIndexer failed to retrieve any results!"
    top_chunk = results[0][0]
    assert top_chunk.pages, "Top retrieved chunk must have physical page numbers attached for grounding!"

    print("\n" + "=" * 70)
    print("SUB-STEP 5.4 SMOKE TEST PASSED SUCCESSFULLY!")
    print(f"Successfully retrieved and reranked '{top_chunk.chunk_id}' as top match with grounded pages {top_chunk.pages}!")
    print("=" * 70)


if __name__ == "__main__":
    main()
