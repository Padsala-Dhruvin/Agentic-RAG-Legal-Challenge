"""
retrieval/legal_hybrid_rag_pipeline.py

Master Concrete Legal RAG Pipeline (Sub-Steps 6.3 - 6.7).
Orchestrates Metadata Indexing (`CaseFactRecord`), Route Resolution (`LegalQuestionRouter`),
Deterministic Bypass, Hybrid Retrieval (`HybridIndexer`), and Dual-Mode Answer Synthesis.
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from arlc.config import EnvConfig, get_config
from retrieval.chunkers import LegalChunk, LegalChunker
from retrieval.free_text_prompts import LEGAL_SYSTEM_PROMPT, build_free_text_prompt
from retrieval.hybrid_rag_pipeline import BaseLegalPipeline
from retrieval.index.hybrid_indexer import HybridIndexer
from retrieval.legal_question_router import LegalQuestionRouter, RoutePlan
from retrieval.loaders.ingested_corpus_loader import LoadedDocument

logger = logging.getLogger(__name__)


@dataclass
class CaseFactRecord:
    """Structured metadata record for deterministic lookups (Sub-Step 6.3)."""
    doc_id: str
    claim_no: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None
    title: Optional[str] = None
    pages_count: int = 0


class LegalHybridRAGPipeline(BaseLegalPipeline):
    """Master end-to-end Legal RAG pipeline implementing items 6.3 through 6.7."""

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__(config)
        self.router = LegalQuestionRouter()
        self.indexer = HybridIndexer(rrf_k=60, config=self.cfg)
        self.chunker = LegalChunker(max_section_chars=4000)
        self.documents: List[LoadedDocument] = []
        self.chunks: List[LegalChunk] = []
        self.fact_records: Dict[str, CaseFactRecord] = {}
        self.llm_client = None
        self.active_model = "local_extractive"

        if not self.cfg.mock_llm:
            try:
                from ingestion.utils import get_llm_client_and_model
                self.llm_client, self.active_model = get_llm_client_and_model(self.cfg)
                logger.info("Connected to cloud LLM provider (%s)", self.active_model)
            except Exception as e:
                logger.warning("Failed to initialize cloud LLM client: %s", e)

    # -------------------------------------------------------------------------
    # Part 1 (6.3): Metadata Indexing & CaseFactRecord
    # -------------------------------------------------------------------------
    def index_corpus(self, documents: List[LoadedDocument]) -> None:
        """Chunk documents, build CaseFactRecords, and populate HybridIndexer."""
        self.documents = documents
        if not documents:
            return

        logger.info("Indexing %d preprocessed legal documents into LegalHybridRAGPipeline...", len(documents))
        self.chunks = self.chunker.chunk_all_documents(documents)
        self.indexer.index_chunks(self.chunks)

        # Build metadata index (`CaseFactRecord`)
        for doc in documents:
            raw_text = "\n".join(b.text for b in doc.blocks)
            claim_match = re.search(r"(?:Claim|Case|Appeal)\s*No[.:]?\s*([A-Z]{2,3}[\s\-]?\d+[\/\-]\d{4}|[A-Z0-9/\-]+)", raw_text, re.IGNORECASE)
            court_match = re.search(r"(?:IN THE\s+)?(DIFC COURTS|COURT OF APPEAL|COURT OF FIRST INSTANCE)", raw_text, re.IGNORECASE)
            date_match = re.search(r"(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b)", raw_text, re.IGNORECASE)

            rec = CaseFactRecord(
                doc_id=doc.doc_id,
                claim_no=claim_match.group(1).strip() if claim_match else None,
                court=court_match.group(1).strip() if court_match else "DIFC Courts",
                date=date_match.group(1).strip() if date_match else None,
                title=doc.doc_id,
                pages_count=doc.metadata.get("page_count", len(doc.blocks)),
            )
            self.fact_records[doc.doc_id.lower()] = rec

        self._is_indexed = True
        logger.info("Successfully indexed %d documents (%d chunks, %d fact records)", len(documents), len(self.chunks), len(self.fact_records))

    # -------------------------------------------------------------------------
    # Part 2 (6.4): Route Resolution & Deterministic Bypass
    # -------------------------------------------------------------------------
    def _try_deterministic_bypass(self, query: str, plan: RoutePlan) -> Optional[Dict[str, Any]]:
        """Instantly resolve exact metadata questions without LLM invocation."""
        q_lower = query.lower()
        if "claim no" in q_lower or "claim number" in q_lower or "case number" in q_lower:
            for doc_id, rec in self.fact_records.items():
                if rec.claim_no:
                    return {
                        "query": query,
                        "answer": f"The claim number is {rec.claim_no} [Doc: {rec.doc_id}, Page: 1].",
                        "citations": [{"doc_id": rec.doc_id, "page": 1}],
                        "metadata": {"route": "deterministic_bypass", "field": "claim_no", "mode": "deterministic_bypass", "chunks_retrieved": 0},
                    }
        if "what date" in q_lower or "when was" in q_lower:
            for doc_id, rec in self.fact_records.items():
                if rec.date:
                    return {
                        "query": query,
                        "answer": f"The order/judgment date is {rec.date} [Doc: {rec.doc_id}, Page: 1].",
                        "citations": [{"doc_id": rec.doc_id, "page": 1}],
                        "metadata": {"route": "deterministic_bypass", "field": "date", "mode": "deterministic_bypass", "chunks_retrieved": 0},
                    }
        return None

    # -------------------------------------------------------------------------
    # Part 4 (6.6): LLM Invocation vs Local Extractive Fallback
    # -------------------------------------------------------------------------
    def _synthesize_answer(self, query: str, chunks: List[LegalChunk]) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate final answer using cloud LLM or local extractive fallback."""
        if not chunks:
            return "Insufficient evidence in retrieved documents to answer precisely.", []

        # Extract citations format from chunks
        citations = []
        for c in chunks[:2]:
            for page in c.pages:
                citations.append({"doc_id": c.doc_id, "page": page})

        # Mode A: Cloud LLM (`OpenAI / Google Gemini / OpenRouter`)
        if self.llm_client and not self.cfg.mock_llm:
            try:
                prompt = build_free_text_prompt(query, chunks)
                response = self.llm_client.chat.completions.create(
                    model=self.active_model,
                    messages=[
                        {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                ans = response.choices[0].message.content.strip()
                return ans, citations
            except Exception as e:
                logger.warning("Cloud LLM synthesis failed (%s). Falling back to local extractive.", e)

        # Mode B: Local Extractive Fallback (Offline Mode)
        top_chunk = chunks[0]
        page_ref = f"[Doc: {top_chunk.doc_id}, Page: {top_chunk.pages[0] if top_chunk.pages else 1}]"

        # Split text into sentences to find the best matching sentence for the query
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_chunk.text) if len(s.strip()) > 15]
        q_tokens = set(re.findall(r"\w+", query.lower()))

        best_sentence = ""
        best_overlap = -1.0
        for s in sentences:
            s_tokens = set(re.findall(r"\w+", s.lower()))
            overlap = len(q_tokens.intersection(s_tokens))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s

        if not best_sentence or best_overlap == 0:
            best_sentence = top_chunk.text[:250].strip() + "..."

        ans = f"{best_sentence} {page_ref}"
        return ans, citations

    # -------------------------------------------------------------------------
    # Part 5 (6.7): Master Pipeline Orchestration
    # -------------------------------------------------------------------------
    def answer_question(self, query: str, **kwargs: Any) -> Dict[str, Any]:
        """Execute full route -> bypass -> retrieve -> fuse -> synthesize pipeline."""
        if not self.is_ready():
            raise RuntimeError("Pipeline not indexed! Call index_corpus() before answer_question().")

        # Step 1: Route resolution
        plan = self.router.route_question(query)
        logger.info("Router classification: route='%s', filter_doc='%s'", plan.route, plan.target_doc_id)

        # Step 2: Check deterministic bypass (`Part 2`)
        if plan.route == "METADATA_LOOKUP" or "claim no" in query.lower() or "what date" in query.lower():
            bypass_res = self._try_deterministic_bypass(query, plan)
            if bypass_res:
                return bypass_res

        # Step 3: Hybrid Retrieval & Fusion (`Part 3`)
        filter_doc = plan.target_doc_id
        filter_rep = plan.extracted_filters.get("representation")
        top_k = kwargs.get("top_k", 4)

        retrieved_hits = self.indexer.search(
            query=query,
            top_k=top_k,
            filter_doc_id=filter_doc,
            filter_representation=filter_rep,
        )

        top_chunks = [chunk for chunk, _ in retrieved_hits]
        top_score = retrieved_hits[0][1] if retrieved_hits else 0.0

        # Step 4: Answer Synthesis (`Part 4`)
        answer_text, citations = self._synthesize_answer(query, top_chunks)

        # Step 5: Packaging & Return (`Part 5`)
        return {
            "query": query,
            "answer": answer_text,
            "citations": citations,
            "metadata": {
                "route": plan.route,
                "top_score": top_score,
                "chunks_retrieved": len(top_chunks),
                "mode": self.active_model if (self.llm_client and not self.cfg.mock_llm) else "local_extractive",
            },
        }
