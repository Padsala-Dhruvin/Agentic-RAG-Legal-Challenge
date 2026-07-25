"""
retrieval/index/hybrid_indexer.py

Hybrid Indexer combining Lexical (BM25Okapi) and Semantic (Dense/TF-IDF Vector) search
via Reciprocal Rank Fusion (RRF, k=60), followed by our active Reranker (`HeuristicReranker`).
Supports both Cloud Mode (`TurbopufferStore` + `Cohere`) and Local/Offline Mode (`InMemoryVectorStore`).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from arlc.config import EnvConfig, get_config
from retrieval.chunkers.legal_chunk_types import LegalChunk
from retrieval.turbopuffer_store import TurbopufferStore
from retrieval.utils.cohere_rate_limit import batch_embed_with_backoff
from retrieval.utils.rerankers import BaseReranker, get_reranker

logger = logging.getLogger(__name__)


class InMemoryVectorStore:
    """Local offline vector store using scikit-learn TF-IDF and cosine similarity."""

    def __init__(self) -> None:
        self.chunks: List[LegalChunk] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
        except Exception as e:
            logger.warning("scikit-learn not available (%s). InMemoryVectorStore will use basic overlap.", e)

    def upsert_chunks(self, chunks: List[LegalChunk]) -> None:
        self.chunks = chunks
        if not chunks or not self.vectorizer:
            return
        texts = [c.text for c in chunks]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info("Fitted local InMemoryVectorStore (TF-IDF matrix shape: %s)", self.tfidf_matrix.shape)

    def query(self, query_text: str, top_k: int = 30) -> List[Tuple[LegalChunk, float]]:
        if not self.chunks:
            return []
        if self.vectorizer and self.tfidf_matrix is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self.vectorizer.transform([query_text])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            scored = list(zip(self.chunks, [float(s) for s in sims]))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]

        # Basic overlap fallback if scikit-learn missing
        q_tokens = set(re.findall(r"\w+", query_text.lower()))
        scored = []
        for c in self.chunks:
            c_tokens = set(re.findall(r"\w+", c.text.lower()))
            overlap = len(q_tokens.intersection(c_tokens)) / max(1, len(q_tokens))
            scored.append((c, float(overlap)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class HybridIndexer:
    """Orchestrates parallel BM25 and Vector indexing with RRF fusion and reranking."""

    def __init__(self, rrf_k: int = 60, config: Optional[EnvConfig] = None) -> None:
        self.rrf_k = rrf_k
        self.cfg = config or get_config()
        self.chunks: List[LegalChunk] = []
        self.bm25 = None
        self.reranker: BaseReranker = get_reranker(self.cfg)

        # Dual-mode vector store setup
        self.cloud_store = TurbopufferStore(config=self.cfg)
        self.local_store = InMemoryVectorStore() if not self.cloud_store.is_active() else None
        self.cohere_client = None

        if self.cfg.cohere_api_key and self.cloud_store.is_active():
            try:
                import cohere
                self.cohere_client = cohere.Client(self.cfg.cohere_api_key)
            except Exception as e:
                logger.warning("Failed to initialize Cohere client: %s", e)

    def index_chunks(self, chunks: List[LegalChunk]) -> None:
        """Build BM25 index and populate active vector store across all chunks."""
        self.chunks = chunks
        if not chunks:
            return

        # 1. Build BM25 index
        try:
            from rank_bm25 import BM25Okapi
            tokenized = [re.findall(r"\w+", c.text.lower()) for c in chunks]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("Fitted BM25Okapi index on %d chunks", len(chunks))
        except Exception as e:
            logger.warning("rank_bm25 not available (%s). BM25 search disabled.", e)

        # 2. Build Vector index (Cloud vs Local)
        if self.cloud_store.is_active() and self.cohere_client:
            texts = [c.text for c in chunks]
            vecs = batch_embed_with_backoff(self.cohere_client, texts)
            self.cloud_store.upsert_chunks(chunks, vecs)
        elif self.local_store:
            self.local_store.upsert_chunks(chunks)

    def _search_bm25(self, query_text: str, top_k: int = 30) -> List[Tuple[LegalChunk, float]]:
        if not self.bm25 or not self.chunks:
            return []
        q_tokens = re.findall(r"\w+", query_text.lower())
        scores = self.bm25.get_scores(q_tokens)
        scored = list(zip(self.chunks, [float(s) for s in scores]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _search_vector(self, query_text: str, top_k: int = 30) -> List[Tuple[LegalChunk, float]]:
        if self.cloud_store.is_active() and self.cohere_client:
            q_vec = self.cohere_client.embed(texts=[query_text], model="embed-english-v3.0", input_type="search_query").embeddings[0]
            return self.cloud_store.query([float(x) for x in q_vec], top_k=top_k)
        elif self.local_store:
            return self.local_store.query(query_text, top_k=top_k)
        return []

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_doc_id: Optional[str] = None,
        filter_representation: Optional[str] = None,
    ) -> List[Tuple[LegalChunk, float]]:
        """Run BM25 + Vector search, apply filters, perform RRF fusion, and rerank top candidates."""
        if not self.chunks:
            return []

        # 1. Retrieve candidates from both engines
        bm25_hits = self._search_bm25(query, top_k=top_k * 3)
        vec_hits = self._search_vector(query, top_k=top_k * 3)

        # 2. Apply metadata filters if specified (`doc_id` or `representation`)
        def _passes_filter(c: LegalChunk) -> bool:
            if filter_doc_id and c.doc_id.lower() != filter_doc_id.lower():
                return False
            if filter_representation and c.representation != filter_representation:
                return False
            return True

        # 3. Calculate Reciprocal Rank Fusion (RRF) scores
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, LegalChunk] = {}

        for rank, (chunk, _) in enumerate(bm25_hits, 1):
            if _passes_filter(chunk):
                chunk_map[chunk.chunk_id] = chunk
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (chunk, _) in enumerate(vec_hits, 1):
            if _passes_filter(chunk):
                chunk_map[chunk.chunk_id] = chunk
                rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort by RRF score to pick top candidate pool for reranking
        fused = [(chunk_map[cid], score) for cid, score in rrf_scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        pool = [c for c, _ in fused[: top_k * 2]]

        # 4. Final Reranking pass (`HeuristicReranker`)
        reranked = self.reranker.rerank(query, pool, top_k=top_k)
        return reranked
