"""
examples/test_turbopuffer_offline.py

Smoke test script for Sub-Step 5.3 (Turbopuffer Vector Store Wrapper).
Verifies that TurbopufferStore gracefully deactivates when TURBOPUFFER_API_KEY is unset,
and tests upsert/query reconstruction logic using a mock namespace object.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.chunkers.legal_chunk_types import LegalChunk
from retrieval.turbopuffer_store import TurbopufferStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_turbopuffer_offline")


@dataclass
class MockQueryResult:
    id: str
    dist: float
    attributes: Dict[str, Any]


class MockNamespace:
    """Mock Turbopuffer namespace verifying upsert attributes and returning simulated query hits."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.stored_ids: List[str] = []
        self.stored_vectors: List[List[float]] = []
        self.stored_attrs: Dict[str, List[Any]] = {}

    def upsert(self, ids: List[str], vectors: List[List[float]], attributes: Dict[str, List[Any]], distance_metric: str = "cosine") -> None:
        logger.info("MockNamespace.upsert called for %d items (metric=%s)", len(ids), distance_metric)
        self.stored_ids = ids
        self.stored_vectors = vectors
        self.stored_attrs = attributes

    def query(self, vector: List[float], top_k: int = 20, filters: Optional[Dict[str, Any]] = None, include_attributes: Optional[List[str]] = None) -> List[MockQueryResult]:
        logger.info("MockNamespace.query called with top_k=%d, filters=%s", top_k, filters)
        results = []
        for idx, cid in enumerate(self.stored_ids[:top_k]):
            attrs = {k: v[idx] for k, v in self.stored_attrs.items() if idx < len(v)}
            results.append(MockQueryResult(id=cid, dist=0.15 * (idx + 1), attributes=attrs))
        return results


def main() -> None:
    logger.info("Initializing smoke test for Sub-Step 5.3 (Turbopuffer Store)...")

    # 1. Test Offline / Inactive Graceful Degradation
    store = TurbopufferStore(namespace_name="test_inactive")
    logger.info("Testing offline initialization without API key...")
    assert not store.is_active(), "Expected store to be inactive when TURBOPUFFER_API_KEY is missing!"
    assert store.query([0.1, 0.2]) == [], "Expected empty query result when inactive!"
    logger.info(" -> Verified offline Graceful Degradation (store.is_active() == False)\n")

    # 2. Test Active Mock Upsert and Query Reconstruction
    logger.info("Testing active upsert and query reconstruction using MockNamespace...")
    store.ns = MockNamespace("test_mock_active")

    sample_chunks = [
        LegalChunk(
            chunk_id="doc_101_sec_0",
            doc_id="doc_101",
            representation="section",
            text="The appellant argued that the tribunal erred in law by refusing the security for costs application.",
            pages=[1, 2],
        ),
        LegalChunk(
            chunk_id="doc_101_sec_1",
            doc_id="doc_101",
            representation="section",
            text="The respondent submitted that costs were already secured under Article 43(2).",
            pages=[3],
        ),
    ]
    sample_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    success = store.upsert_chunks(sample_chunks, sample_vectors)
    assert success, "Expected upsert_chunks to succeed when mock namespace is injected!"

    # Query the mock store
    hits = store.query(query_vector=[0.1, 0.2, 0.3], top_k=2)

    print("-" * 70)
    print(f"Total Query Hits Returned: {len(hits)}")
    for rank, (chunk, score) in enumerate(hits, 1):
        print(f"Hit #{rank} | Chunk ID: {chunk.chunk_id} | Distance: {score:.4f} | Pages: {chunk.pages}")
        print(f"Text: {chunk.text[:80]}...")
        print("-" * 70)

    assert len(hits) == 2, "Expected 2 reconstructed LegalChunks from query!"
    assert hits[0][0].chunk_id == "doc_101_sec_0", "Chunk ID reconstruction mismatch!"
    assert hits[0][0].pages == [1, 2], "Pages attribute reconstruction mismatch!"

    print("\n" + "=" * 70)
    print("SUB-STEP 5.3 SMOKE TEST PASSED SUCCESSFULLY!")
    print("Both offline Graceful Degradation and active upsert/query reconstruction verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
