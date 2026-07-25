"""
examples/test_reranker_offline.py

Smoke test script for Sub-Step 5.1 (Reranker Framework).
Verifies that get_reranker() loads HeuristicReranker and cleanly promotes the true matching chunk to position #1.
"""

import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.chunkers.legal_chunk_types import LegalChunk
from retrieval.utils.rerankers import get_reranker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_reranker_offline")


def main() -> None:
    logger.info("Initializing smoke test for Sub-Step 5.1 (Reranker Framework)...")
    reranker = get_reranker()
    logger.info("Active Reranker Class: %s", reranker.__class__.__name__)

    # Create 3 candidate chunks out of order
    chunks = [
        LegalChunk(
            chunk_id="chunk_1_irrelevant",
            doc_id="case_001",
            representation="section",
            text="The tenant entered into a commercial office lease agreement in Dubai Marina on 15 March 2024.",
            pages=[1],
        ),
        LegalChunk(
            chunk_id="chunk_2_partial",
            doc_id="case_001",
            representation="section",
            text="LXT Real Estate Broker L.L.C acted as the primary agent facilitating property viewings and negotiations.",
            pages=[2],
        ),
        LegalChunk(
            chunk_id="chunk_3_exact_answer",
            doc_id="case_001",
            representation="section",
            text="The order for security will remain in place and assessed and fixed in the amount of AED 550,000, to be paid within 14 days.",
            pages=[3],
        ),
    ]

    query = "What is the assessed amount for security for costs?"
    logger.info("Query: \"%s\"", query)
    logger.info("Initial Order before Reranking: %s", [c.chunk_id for c in chunks])

    reranked = reranker.rerank(query, chunks, top_k=3)

    print("-" * 70)
    for rank, (chunk, score) in enumerate(reranked, 1):
        print(f"Rank #{rank} | Score: {score:.4f} | Chunk ID: {chunk.chunk_id}")
        print(f"Text Preview: {chunk.text[:100]}...")
        print("-" * 70)

    assert reranked[0][0].chunk_id == "chunk_3_exact_answer", "Reranker failed to promote the exact answer chunk to Rank #1!"

    print("\n" + "=" * 70)
    print("SUB-STEP 5.1 SMOKE TEST PASSED SUCCESSFULLY!")
    print(f"Successfully promoted '{reranked[0][0].chunk_id}' to Rank #1 using {reranker.__class__.__name__}!")
    print("=" * 70)


if __name__ == "__main__":
    main()
