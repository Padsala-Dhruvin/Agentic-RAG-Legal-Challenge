"""
examples/test_base_pipeline_offline.py

Smoke test script for Sub-Step 6.2 (Base Pipeline Alias).
Verifies that HybridRAGPipeline initializes, enforces interface contracts, and tracks readiness.
"""

import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import BaseLegalPipeline, HybridRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_base_pipeline_offline")


def main() -> None:
    logger.info("Initializing smoke test for Sub-Step 6.2 (Base Pipeline Alias)...")

    pipeline = HybridRAGPipeline()
    assert isinstance(pipeline, BaseLegalPipeline), "HybridRAGPipeline must inherit from BaseLegalPipeline!"
    assert not pipeline.is_ready(), "Expected pipeline not to be ready before index_corpus is called!"

    logger.info("Testing base index_corpus method...")
    pipeline.index_corpus(documents=[])
    assert pipeline.is_ready(), "Expected pipeline to be ready after index_corpus is called!"

    logger.info("Testing base answer_question contract...")
    res = pipeline.answer_question("What is the security for costs amount?")
    print("-" * 70)
    print("Base Answer Output Dict:")
    for k, v in res.items():
        print(f"  {k}: {v}")
    print("-" * 70)

    assert "query" in res and "answer" in res and "citations" in res, "Base output dict missing required contract keys!"

    print("\n" + "=" * 70)
    print("SUB-STEP 6.2 SMOKE TEST PASSED SUCCESSFULLY!")
    print("BaseLegalPipeline contract and HybridRAGPipeline alias verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
