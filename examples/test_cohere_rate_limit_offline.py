"""
examples/test_cohere_rate_limit_offline.py

Smoke test script for Sub-Step 5.2 (Rate-Limited Cohere Embeddings).
Uses a mock Cohere client that intentionally throws 429 rate limit exceptions
to verify that our retry and batching logic works offline.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.utils.cohere_rate_limit import batch_embed_with_backoff, embed_with_backoff

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_cohere_rate_limit_offline")


@dataclass
class MockCohereResponse:
    embeddings: List[List[float]]


class MockCohereClient:
    """Mock client that fails twice with 429 Rate Limit before succeeding."""

    def __init__(self) -> None:
        self.call_count = 0

    def embed(self, texts: List[str], model: str = "", input_type: str = "") -> MockCohereResponse:
        self.call_count += 1
        logger.info("MockCohereClient.embed called (Attempt #%d with %d texts)", self.call_count, len(texts))
        if self.call_count <= 2:
            raise Exception("HTTP 429 Too Many Requests: Rate limit exceeded on trial key!")
        # Return dummy 3-dimensional embeddings on the 3rd attempt
        dummy_vecs = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(len(texts))]
        return MockCohereResponse(embeddings=dummy_vecs)


def main() -> None:
    logger.info("Initializing smoke test for Sub-Step 5.2 (Cohere Rate Limiter)...")
    mock_client = MockCohereClient()
    sample_texts = ["Legal chunk 1: Security for costs", "Legal chunk 2: DIFC Court of Appeal"]

    logger.info("Testing embed_with_backoff with simulated 429 errors...")
    # Use small initial_backoff=0.1s so the test runs quickly
    vecs = embed_with_backoff(
        client=mock_client,
        texts=sample_texts,
        max_retries=5,
        initial_backoff=0.1,
    )

    print("-" * 70)
    print(f"Total Attempts Required: {mock_client.call_count}")
    print(f"Successfully Recovered Vectors: {len(vecs)} items (Sample: {vecs[0]})")
    print("-" * 70)

    assert mock_client.call_count == 3, "Expected exactly 3 calls (2 retried failures + 1 success)"
    assert len(vecs) == 2, "Expected exactly 2 vector outputs"

    # Test batch slicing
    logger.info("\nTesting batch_embed_with_backoff across 250 dummy texts (batch_size=96)...")
    dummy_client = MockCohereClient()
    dummy_client.call_count = 5  # Start past failures so batching runs smoothly
    batch_vecs = batch_embed_with_backoff(
        client=dummy_client,
        texts=[f"Chunk {i}" for i in range(250)],
        batch_size=96,
        sleep_between_batches=0.05,
    )

    print("-" * 70)
    print(f"Total Batch Outputs Created: {len(batch_vecs)} vectors (from 250 inputs across 3 batches)")
    print("-" * 70)

    assert len(batch_vecs) == 250, "Batch embed failed to return exactly 250 vectors"

    print("\n" + "=" * 70)
    print("SUB-STEP 5.2 SMOKE TEST PASSED SUCCESSFULLY!")
    print("Both exponential backoff recovery and batch slicing verified without errors!")
    print("=" * 70)


if __name__ == "__main__":
    main()
