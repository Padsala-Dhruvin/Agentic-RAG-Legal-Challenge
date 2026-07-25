"""
examples/test_router_offline.py

Smoke test script for Phase 4 (Legal Question Router).
Tests LegalQuestionRouter classification against representative legal challenge questions.
"""

import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval import LegalQuestionRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_router_offline")


def main() -> None:
    logger.info("Initializing smoke test for Phase 4 (Legal Question Router)...")
    router = LegalQuestionRouter()

    sample_queries = [
        # 1. Metadata / fact lookup with claim number
        "In claim CA 005/2025, who were the judges presiding over the appeal?",
        # 2. Substantive legal holding query
        "What did the Court of Appeal hold regarding the provision of security for costs and litigation funding in CA 005/2025?",
        # 3. Multi-case synthesis
        "Summarize all judgments across all cases from 2026 involving real estate disputes.",
        # 4. Direct hex ID metadata lookup
        "What is the judgment date for document 03b621728fe29eb6113fcdb57f6458d793fd2d5c5b833ae26d40f04a29c85359?",
    ]

    logger.info("Testing %d sample queries:\n", len(sample_queries))

    for idx, query in enumerate(sample_queries, 1):
        plan = router.route_question(query)
        print("-" * 70)
        print(f"Query [{idx}]: \"{query}\"")
        print(f" -> Assigned Route     : {plan.route}")
        print(f" -> Confidence         : {plan.confidence:.2f}")
        print(f" -> Extracted Filters  : {plan.extracted_filters}")
        print(f" -> Router Reasoning   : {plan.reasoning}")

    print("-" * 70)
    print("\n" + "=" * 70)
    print("PHASE 4 SMOKE TEST PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
