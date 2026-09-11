"""
retrieval/legal_hybrid_rag_pipeline.py

Master Concrete Legal RAG Pipeline (Sub-Steps 6.3 - 6.7).
Orchestrates Metadata Indexing (`CaseFactRecord`), Route Resolution (`LegalQuestionRouter`),
Deterministic Bypass, Hybrid Retrieval (`HybridIndexer`), and Dual-Mode Answer Synthesis.
"""

import logging
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from arlc.config import EnvConfig, get_config
from arlc.llm_cache import LLMResponseCache
from retrieval.chunkers import LegalChunk, LegalChunker
from retrieval.free_text_prompts import LEGAL_SYSTEM_PROMPT, build_free_text_prompt
from retrieval.hybrid_rag_pipeline import BaseLegalPipeline
from retrieval.index.hybrid_indexer import HybridIndexer
from retrieval.legal_question_router import LegalQuestionRouter, RoutePlan
from retrieval.loaders.ingested_corpus_loader import LoadedDocument
from retrieval.query_rewriter import LegalQueryRewriter

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
        self.query_rewriter = LegalQueryRewriter(enabled=self.cfg.enable_query_rewrite)
        self.documents: List[LoadedDocument] = []
        self.chunks: List[LegalChunk] = []
        self.fact_records: Dict[str, CaseFactRecord] = {}
        self.document_summaries: Dict[str, str] = {}
        self.llm_cache = LLMResponseCache(self.cfg.llm_cache_path)
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
        self.document_summaries = {}
        for doc in documents:
            self.document_summaries[doc.doc_id] = self._build_document_summary(doc)
            doc.metadata["document_summary"] = self.document_summaries[doc.doc_id]
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

    @staticmethod
    def _build_document_summary(doc: LoadedDocument) -> str:
        """Create a deterministic document-level summary for coarse retrieval."""
        first_text = " ".join(block.text for block in doc.blocks[:5]).strip()
        return (
            f"Document {doc.doc_id}. Type: {doc.metadata.get('doc_type', 'unknown')}. "
            f"Title: {doc.metadata.get('official_title') or doc.doc_id}. "
            f"Claim: {doc.metadata.get('claim_number') or 'unknown'}. "
            f"Court: {doc.metadata.get('court') or 'unknown'}. "
            f"Date: {doc.metadata.get('date') or 'unknown'}. "
            f"Opening text: {first_text[:700]}"
        )

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
    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower()))

    def _validate_citations(self, citations: List[Dict[str, Any]], chunks: List[LegalChunk]) -> List[Dict[str, Any]]:
        """Keep only citations that point to retrieved chunks and valid pages."""
        valid_pages = {(chunk.doc_id, page) for chunk in chunks for page in chunk.pages}
        return [
            {"doc_id": cit["doc_id"], "page": cit["page"]}
            for cit in citations
            if cit.get("doc_id", "") and (cit.get("doc_id"), cit.get("page")) in valid_pages
        ]

    def _is_grounded(self, answer: str, chunks: List[LegalChunk]) -> bool:
        """Apply a lightweight lexical groundedness gate before returning an answer."""
        answer_tokens = self._tokens(re.sub(r"\[Doc:.*?\]", "", answer))
        context_tokens = self._tokens(" ".join(chunk.text for chunk in chunks))
        if not answer_tokens:
            return False
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        return overlap >= self.cfg.groundedness_min_overlap

    def _call_llm_with_retry(self, prompt: str) -> str:
        """Call the LLM with a persistent cache and bounded exponential retry."""
        messages = [
            {"role": "system", "content": LEGAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        cache_key = self.llm_cache.key(self.active_model, messages, 0.0)
        cached = self.llm_cache.get(cache_key)
        if cached is not None:
            return cached

        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.cfg.llm_retry_attempts)):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.active_model,
                    messages=messages,
                    temperature=0.0,
                )
                answer = response.choices[0].message.content.strip()
                self.llm_cache.put(cache_key, answer)
                return answer
            except Exception as exc:
                last_error = exc
                if attempt + 1 < max(1, self.cfg.llm_retry_attempts):
                    time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    @staticmethod
    def _parse_structured_response(raw_response: str) -> Optional[Dict[str, Any]]:
        """Parse strict JSON plus common fenced-JSON model formatting."""
        candidates = [raw_response.strip()]
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.insert(0, fenced.group(1))
        object_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if object_match:
            candidates.append(object_match.group(0))

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
                if isinstance(payload, dict):
                    return payload
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        return None

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
                raw_response = self._call_llm_with_retry(prompt)
                try:
                    payload = self._parse_structured_response(raw_response)
                    if payload is None:
                        raise ValueError("No JSON object found")
                    ans = str(payload.get("answer", "")).strip()
                    model_citations = payload.get("citations", [])
                    validated = self._validate_citations(model_citations, chunks)
                    if not payload.get("supported", True) or not ans or not self._is_grounded(ans, chunks):
                        return "Insufficient evidence in retrieved documents to answer precisely.", validated or citations
                    return ans, validated or citations
                except (TypeError, ValueError, json.JSONDecodeError):
                    logger.warning("LLM returned non-JSON output; applying groundedness fallback.")
                    if self._is_grounded(raw_response, chunks):
                        return raw_response, citations
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
        if not self._is_grounded(ans, chunks):
            return "Insufficient evidence in retrieved documents to answer precisely.", citations
        return ans, citations

    # -------------------------------------------------------------------------
    # Part 5 (6.7): Master Pipeline Orchestration
    # -------------------------------------------------------------------------
    def _merge_retrieval_hits(self, all_hits: List[List[Tuple[LegalChunk, float]]], top_k: int) -> List[Tuple[LegalChunk, float]]:
        """Fuse retrieval hits from query variants using Reciprocal Rank Fusion."""
        rrf_k = self.indexer.rrf_k
        by_chunk: Dict[str, Tuple[LegalChunk, float]] = {}

        for hits in all_hits:
            for rank, (chunk, _) in enumerate(hits, 1):
                current_chunk, current_score = by_chunk.get(chunk.chunk_id, (chunk, 0.0))
                fused_score = current_score + (1.0 / (rrf_k + rank))
                by_chunk[chunk.chunk_id] = (current_chunk, fused_score)

        merged = list(by_chunk.values())
        merged.sort(key=lambda x: x[1], reverse=True)
        return merged[:top_k]

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

        # Query rewriting is intentionally applied only to substantive section search.
        search_queries = self.query_rewriter.generate_candidates(query, route=plan.route)
        all_hits: List[List[Tuple[LegalChunk, float]]] = []
        for q in search_queries:
            hits = self.indexer.search(
                query=q,
                top_k=top_k,
                filter_doc_id=filter_doc,
                filter_representation=filter_rep,
            )
            all_hits.append(hits)

        retrieved_hits = self._merge_retrieval_hits(all_hits, top_k=top_k)

        if retrieved_hits and retrieved_hits[0][1] < self.cfg.retrieval_min_score:
            retrieved_hits = []

        if retrieved_hits and self.cfg.enable_final_rerank:
            candidate_chunks = [chunk for chunk, _ in retrieved_hits]
            reranked = self.indexer.reranker.rerank(query, candidate_chunks, top_k=top_k)
            if reranked:
                retrieved_hits = reranked

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
                "search_queries": search_queries,
                "mode": self.active_model if (self.llm_client and not self.cfg.mock_llm) else "local_extractive",
            },
        }
