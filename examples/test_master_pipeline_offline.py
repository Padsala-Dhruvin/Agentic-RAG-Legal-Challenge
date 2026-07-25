"""
examples/test_master_pipeline_offline.py

Smoke test script for Sub-Steps 6.3 to 6.7 (LegalHybridRAGPipeline).
Verifies exact deterministic bypass for claim numbers (`Part 2`),
and end-to-end hybrid RAG retrieval + local extractive grounded synthesis (`Part 3 & 4`).
"""

import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import IngestedCorpusLoader, LegalHybridRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_master_pipeline_offline")


def main() -> None:
    logger.info("Initializing smoke test for Sub-Steps 6.3 - 6.7 (LegalHybridRAGPipeline)...")

    ingest_dir = Path("ingestion/test_ingest_output")
    if not ingest_dir.exists():
        logger.error("Preprocessed docs missing. Run Phase 2 first.")
        sys.exit(1)

    loader = IngestedCorpusLoader(ingest_dir=ingest_dir)
    docs = loader.load_all_documents()

    # 1. Initialize and Index Corpus (`Part 1 - 6.3`)
    pipeline = LegalHybridRAGPipeline()
    pipeline.index_corpus(docs)
    assert pipeline.is_ready(), "Expected pipeline.is_ready() == True after index_corpus!"

    # 2. Test Deterministic Bypass (`Part 2 - 6.4`)
    logger.info("\nTesting Deterministic Bypass for Claim Number...")
    bypass_query = "What is the claim number for this appeal?"
    res_bypass = pipeline.answer_question(bypass_query)

    print("-" * 70)
    print("DETERMINISTIC BYPASS RESULT:")
    print(f"  Query: \"{res_bypass['query']}\"")
    print(f"  Answer: \"{res_bypass['answer']}\"")
    print(f"  Citations: {res_bypass['citations']}")
    print(f"  Route Used: {res_bypass['metadata']['route']}")
    print("-" * 70)

    assert res_bypass["metadata"]["route"] == "deterministic_bypass", "Expected deterministic bypass route!"
    assert "CA 005/2025" in res_bypass["answer"], "Failed to extract exact claim number CA 005/2025!"

    # 3. Test Full Hybrid RAG Synthesis (`Part 3 & 4 - 6.5 & 6.6`)
    logger.info("\nTesting Full Hybrid RAG Retrieval + Extractive Synthesis...")
    rag_query = "What did the Court order regarding the AED 550,000 security for costs?"
    res_rag = pipeline.answer_question(rag_query)

    print("-" * 70)
    print("HYBRID RAG SYNTHESIS RESULT:")
    print(f"  Query: \"{res_rag['query']}\"")
    print(f"  Answer:\n    \"{res_rag['answer']}\"")
    print(f"  Citations: {res_rag['citations'][:3]}")
    print(f"  Route Used: {res_rag['metadata']['route']}")
    print(f"  Synthesis Mode: {res_rag['metadata']['mode']}")
    print("-" * 70)

    assert "AED 550,000" in res_rag["answer"], "Failed to synthesize answer with AED 550,000!"
    assert res_rag["citations"], "Must include physical page citations attached to answer!"

    print("\n" + "=" * 70)
    print("SUB-STEPS 6.3 - 6.7 SMOKE TEST PASSED SUCCESSFULLY!")
    print("Both Deterministic Bypass AND Hybrid Extractive RAG verified with exact citations!")
    print("=" * 70)


if __name__ == "__main__":
    main()
