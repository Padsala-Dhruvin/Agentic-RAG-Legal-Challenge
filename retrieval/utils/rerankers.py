"""
retrieval/utils/rerankers.py

Reranker Framework (Sub-Step 5.1).
Provides unified re-ranking of retrieved chunks using:
1. CohereReranker (Cloud: rerank-v3-512)
2. VoyageReranker (Cloud: rerank-2)
3. LocalMiniLMReranker (Local PyTorch CrossEncoder if installed)
4. HeuristicReranker (Pure Python exact-phrase & overlap scoring - Default Offline Choice)
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from arlc.config import EnvConfig, get_config
from retrieval.chunkers.legal_chunk_types import LegalChunk

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract interface for candidate chunk rerankers."""

    @abstractmethod
    def rerank(self, query: str, chunks: List[LegalChunk], top_k: int = 5) -> List[Tuple[LegalChunk, float]]:
        """Rerank chunks and return top_k (chunk, score) tuples."""
        pass


class HeuristicReranker(BaseReranker):
    """Pure-Python exact-phrase and word overlap reranker requiring 0 dependencies or API keys."""

    def rerank(self, query: str, chunks: List[LegalChunk], top_k: int = 5) -> List[Tuple[LegalChunk, float]]:
        if not chunks:
            return []

        q_tokens = set(re.findall(r"\w+", query.lower()))
        scored = []
        for chunk in chunks:
            c_text_lower = chunk.text.lower()
            # Check exact query phrase match boost (+2.0 points)
            phrase_boost = 2.0 if query.lower().strip() in c_text_lower else 0.0
            c_tokens = set(re.findall(r"\w+", c_text_lower))
            overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
            score = overlap + phrase_boost
            scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# class LocalMiniLMReranker(BaseReranker):
#     """Local neural cross-encoder using sentence_transformers (PyTorch)."""

#     def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
#         self.model_name = model_name
#         self.model = None
#         try:
#             from sentence_transformers import CrossEncoder
#             self.model = CrossEncoder(model_name)
#             logger.info("Loaded local CrossEncoder model: %s", model_name)
#         except Exception as e:
#             logger.debug("sentence_transformers not available (%s).", e)

#     def rerank(self, query: str, chunks: List[LegalChunk], top_k: int = 5) -> List[Tuple[LegalChunk, float]]:
#         if not chunks:
#             return []

#         if self.model is not None:
#             pairs = [[query, c.text] for c in chunks]
#             scores = self.model.predict(pairs)
#             scored = list(zip(chunks, [float(s) for s in scores]))
#             scored.sort(key=lambda x: x[1], reverse=True)
#             return scored[:top_k]

#         logger.info("LocalMiniLM model not loaded. Falling back to HeuristicReranker.")
#         return HeuristicReranker().rerank(query, chunks, top_k)


# class CohereReranker(BaseReranker):
#     """Cloud reranker using Cohere API."""

#     def __init__(self, api_key: str, model: str = "rerank-v3-512") -> None:
#         self.api_key = api_key
#         self.model = model
#         self.client = None
#         try:
#             import cohere
#             self.client = cohere.Client(api_key)
#             logger.info("Initialized Cohere Reranker (%s)", model)
#         except Exception as e:
#             logger.warning("Failed to initialize Cohere SDK: %s", e)

#     def rerank(self, query: str, chunks: List[LegalChunk], top_k: int = 5) -> List[Tuple[LegalChunk, float]]:
#         if not chunks or not self.client:
#             return []
#         try:
#             docs = [c.text for c in chunks]
#             response = self.client.rerank(query=query, documents=docs, top_n=top_k, model=self.model)
#             results = []
#             for item in response.results:
#                 results.append((chunks[item.index], float(item.relevance_score)))
#             return results
#         except Exception as e:
#             logger.error("Cohere rerank failed (%s). Falling back to HeuristicReranker.", e)
#             return HeuristicReranker().rerank(query, chunks, top_k)


# class VoyageReranker(BaseReranker):
#     """Cloud reranker using Voyage AI API."""

#     def __init__(self, api_key: str, model: str = "rerank-2") -> None:
#         self.api_key = api_key
#         self.model = model
#         self.client = None
#         try:
#             import voyageai
#             self.client = voyageai.Client(api_key=api_key)
#             logger.info("Initialized Voyage Reranker (%s)", model)
#         except Exception as e:
#             logger.warning("Failed to initialize Voyage SDK: %s", e)

#     def rerank(self, query: str, chunks: List[LegalChunk], top_k: int = 5) -> List[Tuple[LegalChunk, float]]:
#         if not chunks or not self.client:
#             return []
#         try:
#             docs = [c.text for c in chunks]
#             response = self.client.rerank(query=query, documents=docs, model=self.model, top_k=top_k)
#             results = []
#             for item in response.results:
#                 results.append((chunks[item.index], float(item.relevance_score)))
#             return results
#         except Exception as e:
#             logger.error("Voyage rerank failed (%s). Falling back to HeuristicReranker.", e)
#             return HeuristicReranker().rerank(query, chunks, top_k)


def get_reranker(config: Optional[EnvConfig] = None) -> BaseReranker:
    """Factory function selecting the best available reranker based on config/keys."""
    cfg = config or get_config()
    # if cfg.cohere_api_key:
    #     return CohereReranker(api_key=cfg.cohere_api_key)
    # if cfg.voyage_api_key:
    #     return VoyageReranker(api_key=cfg.voyage_api_key)
    # Default offline choice requiring 0 dependencies or API keys:
    return HeuristicReranker()
