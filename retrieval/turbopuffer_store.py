"""
retrieval/turbopuffer_store.py

Turbopuffer Remote Vector Backend (Sub-Step 5.3).
Provides cloud-native vector storage and similarity search for LegalChunks
when operating in Cloud Mode (`TURBOPUFFER_API_KEY`).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from arlc.config import EnvConfig, get_config
from retrieval.chunkers.legal_chunk_types import LegalChunk

logger = logging.getLogger(__name__)


class TurbopufferStore:
    """Remote vector store wrapper using the Turbopuffer serverless engine."""

    def __init__(self, namespace_name: str = "legal_hybrid_rag_corpus", config: Optional[EnvConfig] = None) -> None:
        self.namespace_name = namespace_name
        self.cfg = config or get_config()
        self.ns = None

        if not self.cfg.turbopuffer_api_key:
            logger.debug("TURBOPUFFER_API_KEY not set. TurbopufferStore inactive.")
            return

        try:
            import turbopuffer as tpuf
            tpuf.api_key = self.cfg.turbopuffer_api_key
            self.ns = tpuf.Namespace(namespace_name)
            logger.info("Connected to Turbopuffer namespace: '%s'", namespace_name)
        except Exception as e:
            logger.warning("Failed to initialize Turbopuffer SDK: %s", e)

    def is_active(self) -> bool:
        """Check if connection to Turbopuffer is active and authenticated."""
        return self.ns is not None

    def upsert_chunks(self, chunks: List[LegalChunk], vectors: List[List[float]], distance_metric: str = "cosine") -> bool:
        """Upload chunks and their dense vectors into the Turbopuffer namespace."""
        if not self.is_active() or not chunks or len(chunks) != len(vectors):
            return False

        ids = [c.chunk_id for c in chunks]
        attributes: Dict[str, List[Any]] = {
            "doc_id": [c.doc_id for c in chunks],
            "representation": [c.representation for c in chunks],
            "text": [c.text for c in chunks],
            "pages": [c.pages for c in chunks],
        }

        try:
            self.ns.upsert(
                ids=ids,
                vectors=vectors,
                attributes=attributes,
                distance_metric=distance_metric,
            )
            logger.info("Upserted %d chunks to Turbopuffer namespace '%s'", len(chunks), self.namespace_name)
            return True
        except Exception as e:
            logger.error("Turbopuffer upsert failed: %s", e)
            return False

    def query(self, query_vector: List[float], top_k: int = 20, filters: Optional[Dict[str, Any]] = None) -> List[Tuple[LegalChunk, float]]:
        """Perform similarity search on Turbopuffer and reconstruct LegalChunks."""
        if not self.is_active() or not query_vector:
            return []

        try:
            results = self.ns.query(
                vector=query_vector,
                top_k=top_k,
                filters=filters,
                include_attributes=["doc_id", "representation", "text", "pages"],
            )

            scored_chunks = []
            for item in results:
                attrs = item.attributes or {}
                chunk = LegalChunk(
                    chunk_id=str(item.id),
                    doc_id=attrs.get("doc_id", ""),
                    representation=attrs.get("representation", "section"),
                    text=attrs.get("text", ""),
                    pages=attrs.get("pages", [1]),
                    metadata={"turbopuffer_dist": item.dist},
                )
                scored_chunks.append((chunk, float(item.dist)))

            return scored_chunks
        except Exception as e:
            logger.error("Turbopuffer query failed: %s", e)
            return []
