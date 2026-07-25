"""
examples/legal_hybrid_rag.py

Master Offline Evaluation & Benchmarking Runner (`Sub-Step 7.2`).
Runs the complete LegalHybridRAGPipeline over official benchmark questions (`questions.json`),
tracks sub-millisecond telemetry, and exports the competition `submission.json` package.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arlc.config import get_config
from arlc.submission import SubmissionBuilder
from arlc.telemetry import RetrievalRef, Telemetry, TelemetryTimer, UsageMetrics
from retrieval import IngestedCorpusLoader, LegalHybridRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("legal_hybrid_rag_runner")


def load_questions(questions_path: str) -> List[Dict[str, Any]]:
    """Load evaluation questions from JSON file."""
    p = Path(questions_path)
    if not p.exists():
        logger.error("Questions file not found at %s", questions_path)
        return []
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("questions", [])


def run_evaluation(pipeline: LegalHybridRAGPipeline, questions: List[Dict[str, Any]]) -> SubmissionBuilder:
    """Execute evaluation loop over all questions, tracking telemetry and citations."""
    builder = SubmissionBuilder(architecture_summary="Legal Hybrid RAG (BM25 + TF-IDF RRF + Reranker + Deterministic Bypass + Dual-Mode Synthesis)")

    logger.info("Starting evaluation across %d benchmark questions...", len(questions))
    for idx, q_data in enumerate(questions, 1):
        q_id = str(q_data.get("question_id") or q_data.get("id") or f"q_{idx}")
        q_text = str(q_data.get("question_text") or q_data.get("question") or "")

        logger.info("[%d/%d] Processing Question ID %s: \"%s...\"", idx, len(questions), q_id, q_text[:50])

        with TelemetryTimer() as timer:
            res = pipeline.answer_question(q_text)
            timer.mark_token()
            timing = timer.finish(output_token_count=len(res["answer"].split()))

        # Convert citations to 1-based physical page RetrievalRef objects
        retrieval_refs: List[RetrievalRef] = []
        for cit in res.get("citations", []):
            doc_id = cit.get("doc_id", "")
            page_num = cit.get("page", 1)
            if doc_id:
                retrieval_refs.append(RetrievalRef(doc_id=doc_id, page_numbers=[page_num]))

        usage = UsageMetrics(
            input_tokens=len(q_text.split()) * 2 + 100,
            output_tokens=len(res["answer"].split()) * 2,
        )

        telemetry = Telemetry(
            timing=timing,
            retrieval=retrieval_refs,
            usage=usage,
            model_name=res["metadata"].get("mode", "local_extractive"),
        )

        builder.add_answer(
            question_id=q_id,
            answer=res["answer"],
            telemetry=telemetry,
        )

    return builder


def main() -> None:
    cfg = get_config()
    logger.info("Initializing Master Legal Hybrid RAG Evaluation Runner...")

    # 1. Load preprocessed corpus
    ingest_dir = Path("ingestion/test_ingest_output")
    if not ingest_dir.exists():
        logger.error("Preprocessed corpus missing at %s. Run Phase 2 first.", ingest_dir)
        sys.exit(1)

    loader = IngestedCorpusLoader(ingest_dir=ingest_dir)
    docs = loader.load_all_documents()

    # 2. Index corpus into master pipeline
    pipeline = LegalHybridRAGPipeline(config=cfg)
    pipeline.index_corpus(docs)

    # 3. Load benchmark questions
    questions = load_questions(cfg.questions_path)
    if not questions:
        logger.warning("No questions loaded from %s. Using default smoke questions for verification.", cfg.questions_path)
        questions = [
            {"id": "DIFC_Q01", "question": "What is the claim number for this appeal?"},
            {"id": "DIFC_Q02", "question": "What did the Court order regarding the AED 550,000 security for costs?"},
        ]

    # 4. Run evaluation
    submission_builder = run_evaluation(pipeline, questions)

    # 5. Save submission output
    sub_path = submission_builder.save(cfg.submission_path)
    logger.info("Successfully generated competition submission package: %s", sub_path)

    # 6. Save summary evaluation report
    report_path = Path(cfg.output_report_path)
    report_data = {
        "total_questions": len(submission_builder.answers),
        "submission_file": str(sub_path),
        "architecture": submission_builder.architecture_summary,
        "sample_answers": [a.to_dict() for a in submission_builder.answers[:2]],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("Successfully generated evaluation report: %s", report_path)

    print("\n" + "=" * 70)
    print("PHASE 7.2 BENCHMARK RUNNER COMPLETED SUCCESSFULLY!")
    print(f"Total Questions Evaluated: {len(submission_builder.answers)}")
    print(f"Submission Package Saved:  {sub_path}")
    print(f"Evaluation Report Saved:   {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
