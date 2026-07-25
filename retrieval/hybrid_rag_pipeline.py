"""
retrieval/hybrid_rag_pipeline.py

Base Legal RAG Pipeline Interface and Alias (Sub-Step 6.2).
Defines the abstract contract (`BaseLegalPipeline` and `HybridRAGPipeline`)
that any concrete legal RAG pipeline must satisfy for Phase 7 evaluation and Phase 8 submission.
"""

import abc
import logging
from typing import Any, Dict, List, Optional

from arlc.config import EnvConfig, get_config
from retrieval.loaders.ingested_corpus_loader import LoadedDocument

logger = logging.getLogger(__name__)


class BaseLegalPipeline(abc.ABC):
    """Abstract base class establishing mandatory methods for all legal RAG pipelines."""

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.cfg = config or get_config()
        self._is_indexed = False

    @abc.abstractmethod
    def index_corpus(self, documents: List[LoadedDocument]) -> None:
        """Index preprocessed legal documents into active retrieval engines."""
        pass

    @abc.abstractmethod
    def answer_question(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Process a user question and return structured answer dict with grounded citations."""
        pass

    def is_ready(self) -> bool:
        """Return True if corpus has been indexed and pipeline is ready to answer questions."""
        return self._is_indexed


class HybridRAGPipeline(BaseLegalPipeline):
    """Base alias and foundation class for Hybrid RAG legal pipelines."""

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__(config)
        logger.info("Initialized base HybridRAGPipeline (active model: %s)", self.cfg.llm_model)

    def index_corpus(self, documents: List[LoadedDocument]) -> None:
        logger.warning("index_corpus called on base HybridRAGPipeline. Must be overridden by subclass.")
        self._is_indexed = True

    def answer_question(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        logger.warning("answer_question called on base HybridRAGPipeline. Must be overridden by subclass.")
        return {
            "query": query,
            "answer": "Base HybridRAGPipeline active. Subclass must implement exact synthesis logic.",
            "citations": [],
            "metadata": {},
        }
