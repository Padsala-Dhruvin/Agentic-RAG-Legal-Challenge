# API Reference (`API.md`)

This document provides a technical reference for the public modules, classes, and methods exposed by the **Agentic RAG Legal Challenge** repository.

---

## 1. Challenge Client Library (`arlc/`)

### `arlc.config.EnvConfig`
Dataclass holding all runtime environment configurations loaded from `.env`.
- `docs_dir: str` — Path to input legal PDFs (`docs_corpus`).
- `questions_path: str` — Path to evaluation benchmark file (`questions.json`).
- `submission_path: str` — Output path for generated competition payload (`submission.json`).
- `mock_llm: bool` — When `True`, bypasses network calls for simulated metadata and local synthesis.

### `arlc.telemetry.TelemetryTimer`
High-precision context manager tracking latency and token metrics.
```python
with TelemetryTimer() as timer:
    res = pipeline.answer_question("query")
    timer.mark_token()  # Marks arrival of first token for TTFT calculation
    metrics = timer.finish(output_token_count=10)
# Returns TimingMetrics(ttft_ms=..., tpot_ms=..., total_time_ms=...)
```

### `arlc.submission.SubmissionBuilder`
Stateful builder accumulating answer records and telemetry payloads into `submission.json`.
- `add_answer(question_id: str, answer: Any, telemetry: Optional[Telemetry] = None) -> None`
- `to_dict() -> Dict[str, Any]` — Returns the exact schema required by platform evaluators.
- `save(output_path: Union[str, Path]) -> Path` — Serializes with clean UTF-8 formatting.

---

## 2. Ingestion Engine (`ingestion/`)

### `ingestion.ingest.Ingestion`
Orchestrates offline 3-stage ingestion across PDF files in `docs_dir`.
- `parse(pdf_path: Path, doc_id: str) -> List[Dict[str, Any]]` — Extracts raw text blocks and 0-based `page_idx`.
- `structure_analysis(doc_id: str, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]` — Assigns hierarchical levels (`1=title`, `2=chapter`, `3=section`, `4=body`).
- `metadata_extraction(doc_id: str, content_list: List[Dict[str, Any]]) -> Dict[str, Any]` — Extracts schema-validated case facts (`claim_no`, `court`, `date`).
- `ingest() -> Dict[str, Any]` — Runs the complete pipeline and writes artifacts to `output_dir/`.

---

## 3. Core RAG Architecture (`retrieval/`)

### `retrieval.loaders.IngestedCorpusLoader`
Reads offline ingestion artifacts (`_content_list.json`, `_structure.json`, `_metadata.json`) from disk.
- `load_document(doc_id: str) -> LoadedDocument`
- `load_all_documents() -> List[LoadedDocument]`

### `retrieval.chunkers.LegalChunker`
Segments preprocessed `LoadedDocument` instances into specialized legal chunks.
- `chunk_document(doc: LoadedDocument) -> List[LegalChunk]` — Outputs semantic `title_page`, `section` (up to `max_section_chars`), and `page_anchor` chunks.

### `retrieval.legal_question_router.LegalQuestionRouter`
Analyzes query intent and returns a structured `RoutePlan`.
- `route_question(query: str) -> RoutePlan` — Classifies route into `'METADATA_LOOKUP'`, `'SECTION_SEARCH'`, or `'GLOBAL_SYNTHESIS'`, extracting target document and claim number citations.

### `retrieval.index.hybrid_indexer.HybridIndexer`
Fuses sparse and dense representations for maximum recall and precision.
- `index_chunks(chunks: List[LegalChunk]) -> None` — Fits `BM25Okapi` and local `InMemoryVectorStore` (`TF-IDF`).
- `search(query: str, top_k: int = 4, filter_doc_id: Optional[str] = None, filter_representation: Optional[str] = None) -> List[Tuple[LegalChunk, float]]` — Performs Reciprocal Rank Fusion (`RRF k=60`).

### `retrieval.legal_hybrid_rag_pipeline.LegalHybridRAGPipeline`
Master end-to-end Legal RAG pipeline (`Phase 6`).
- `index_corpus(documents: List[LoadedDocument]) -> None` — Chunks documents, fits indices, and indexes `CaseFactRecord` metadata.
- `answer_question(query: str, **kwargs: Any) -> Dict[str, Any]` — Orchestrates route classification, deterministic metadata bypass (`0ms`), hybrid retrieval, and dual-mode answer synthesis (`cloud LLM vs local extractive`).
