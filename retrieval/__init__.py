"""
retrieval/__init__.py

Top-level retrieval package exporting loaders and chunkers.
"""

from retrieval.loaders import IngestedCorpusLoader, LoadedDocument, TextBlock
from retrieval.chunkers import BaseChunker, LegalChunk, LegalChunker, RecursiveChunker
from retrieval.legal_question_router import LegalQuestionRouter, RoutePlan
from retrieval.turbopuffer_store import TurbopufferStore
from retrieval.index import HybridIndexer, InMemoryVectorStore
from retrieval.free_text_prompts import LEGAL_SYSTEM_PROMPT, build_free_text_prompt
from retrieval.hybrid_rag_pipeline import BaseLegalPipeline, HybridRAGPipeline
from retrieval.legal_hybrid_rag_pipeline import CaseFactRecord, LegalHybridRAGPipeline

__all__ = [
    "IngestedCorpusLoader",
    "LoadedDocument",
    "TextBlock",
    "BaseChunker",
    "LegalChunk",
    "LegalChunker",
    "RecursiveChunker",
    "LegalQuestionRouter",
    "RoutePlan",
    "TurbopufferStore",
    "HybridIndexer",
    "InMemoryVectorStore",
    "LEGAL_SYSTEM_PROMPT",
    "build_free_text_prompt",
    "BaseLegalPipeline",
    "HybridRAGPipeline",
    "CaseFactRecord",
    "LegalHybridRAGPipeline",
]
