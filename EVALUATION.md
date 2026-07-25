# Evaluation & Benchmarking Methodology (`EVALUATION.md`)

This document details the quantitative evaluation framework, scoring formulas, latency benchmarks, and verification results of the **Agentic RAG Legal Challenge** repository.

---

## 1. Competition Scoring Formula

The benchmark evaluation engine computes overall system performance using a weighted composite formula:

$$\text{Final Score} = \left(0.7 \times \text{Accuracy}_{\text{Deterministic}} + 0.3 \times \text{Accuracy}_{\text{LLM Judge}}\right) \times \text{Grounding Multiplier} \times \text{Telemetry Factor} \times \text{TTFT Factor}$$

### Metric Definitions:
1. **$\text{Accuracy}_{\text{Deterministic}}$ (0.7 Weight)**:
   - Exact string matching for dates (`YYYY-MM-DD`), claim numbers (`CA 005/2025`), boolean answers (`true/false`), and numerical statutory amounts (`AED 550,000`).
2. **$\text{Accuracy}_{\text{LLM Judge}}$ (0.3 Weight)**:
   - Evaluated using **RAGAS** (`Faithfulness` and `Answer Relevance`) across complex free-text legal summaries.
3. **$\text{Grounding Multiplier}$ (Crucial Guardrail)**:
   - Strict verification against physical 1-based PDF page citations (`[Doc: doc_id, Page: X]`).
   - If a citation points to a page where the claim text or amount does not exist physically, the grounding multiplier drops towards `0.0`.
4. **$\text{TTFT Factor}$ (Time-To-First-Token Telemetry)**:
   - Penalizes slow initial response latencies (`TTFT > 2000ms`). Our system achieves `TTFT < 50ms` using local extractive synthesis and `0ms` via deterministic bypass.

---

## 2. Telemetry & Citation Normalization

All generated answers must attach a clean `Telemetry` payload via `arlc.telemetry.Telemetry`:
```json
{
  "timing": {
    "ttft_ms": 12.5,
    "tpot_ms": 4.2,
    "total_time_ms": 38.0
  },
  "retrieval": {
    "retrieved_chunk_pages": [
      {
        "doc_id": "03b621728fe29eb6113fcdb57f6458d793fd2d5c5b833ae26d40f04a29c85359",
        "page_numbers": [1, 2, 3]
      }
    ]
  },
  "usage": {
    "input_tokens": 142,
    "output_tokens": 64
  },
  "model_name": "local_extractive"
}
```

### Automatic Normalization (`normalize_retrieved_pages`)
To ensure compliance with platform evaluators, `normalize_retrieved_pages()` automatically:
- Merges duplicate `doc_id` entries.
- Strips non-positive or 0-based page indexes.
- Deduplicates and sorts 1-based physical page numbers in ascending order.

---

## 3. Test Suite Verification & Results (`Phase 7.1`)

Our master unit and integration test suite (`tests/test_legal_hybrid_rag.py`) automatically verifies all core subsystems across 3 standardized test classes:

| Test Class | Asserts | Status |
| :--- | :--- | :--- |
| `Test01_TelemetryAndSubmission` | Sub-millisecond `TelemetryTimer` metrics (`ttft_ms`, `total_time_ms`) & schema-compliant `SubmissionBuilder` dictionary output. | ✅ **PASSED** |
| `Test02_CorpusLoaderAndChunking` | Offline block parsing via `IngestedCorpusLoader` & exact multi-tier chunk generation (`title_page`, `section`, `page_anchor`). | ✅ **PASSED** |
| `Test03_LegalHybridPipeline` | Route classification accuracy (`METADATA_LOOKUP` vs `SECTION_SEARCH`), `0ms` Deterministic Bypass for claim numbers, and physical page-grounded Extractive RAG synthesis. | ✅ **PASSED** |

**Execution Summary:**
```text
.....
----------------------------------------------------------------------
Ran 5 tests in 2.034s

OK
```

---

## 4. Running Benchmark Evaluation Locally (`Phase 7.2`)

To execute the full offline evaluation runner against the official dataset and generate both `submission.json` and `evaluation_report.json`:

```powershell
python examples/legal_hybrid_rag.py
# Or using our master CLI entry point:
python main.py --eval
```
