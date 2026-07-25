"""
tests/test_legal_hybrid_rag.py

Master Unit & Integration Test Suite for the Agentic RAG Legal Challenge (Phase 7.1).
Validates Telemetry, Submission Packaging, Ingestion Loading, Chunking, Routing, and Master Pipeline Synthesis.
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arlc.submission import SubmissionBuilder
from arlc.telemetry import TelemetryTimer
from retrieval import IngestedCorpusLoader, LegalChunker, LegalHybridRAGPipeline, LegalQuestionRouter


class Test01_TelemetryAndSubmission(unittest.TestCase):
    """Test ARLC Challenge Client utilities (`Phase 1 & 3`)."""

    def test_telemetry_timer(self):
        with TelemetryTimer() as timer:
            timer.mark_token()
            metrics = timer.finish(output_token_count=10)
        self.assertGreaterEqual(metrics.ttft_ms, 0.0)
        self.assertGreaterEqual(metrics.total_time_ms, 0.0)

    def test_submission_builder(self):
        builder = SubmissionBuilder(architecture_summary="Test Architecture")
        builder.add_answer(
            question_id="q_001",
            answer="AED 550,000 [Doc: case_001, Page: 3]",
        )
        data = builder.to_dict()
        self.assertEqual(data["architecture_summary"], "Test Architecture")
        self.assertEqual(len(data["answers"]), 1)


class Test02_CorpusLoaderAndChunking(unittest.TestCase):
    """Test offline document loading and legal chunking (`Phase 2 & 4`)."""

    def setUp(self):
        self.ingest_dir = Path("ingestion/test_ingest_output")

    def test_loader_and_chunker(self):
        if not self.ingest_dir.exists():
            self.skipTest("Preprocessed test docs not found in ingestion/test_ingest_output")
        loader = IngestedCorpusLoader(ingest_dir=self.ingest_dir)
        docs = loader.load_all_documents()
        self.assertGreaterEqual(len(docs), 1)

        chunker = LegalChunker(max_section_chars=4000)
        chunks = chunker.chunk_all_documents(docs)
        self.assertGreaterEqual(len(chunks), 1)
        rep_types = {c.representation for c in chunks}
        self.assertIn("section", rep_types)


class Test03_LegalHybridPipeline(unittest.TestCase):
    """Test end-to-end Master Legal Hybrid RAG Pipeline (`Phase 6`)."""

    @classmethod
    def setUpClass(cls):
        cls.ingest_dir = Path("ingestion/test_ingest_output")
        if cls.ingest_dir.exists():
            loader = IngestedCorpusLoader(ingest_dir=cls.ingest_dir)
            cls.docs = loader.load_all_documents()
            cls.pipeline = LegalHybridRAGPipeline()
            cls.pipeline.index_corpus(cls.docs)
        else:
            cls.docs = []
            cls.pipeline = None

    def test_pipeline_ready_and_routing(self):
        if not self.pipeline:
            self.skipTest("Preprocessed test docs missing")
        self.assertTrue(self.pipeline.is_ready())

        router = LegalQuestionRouter()
        plan_meta = router.route_question("What is the claim number?")
        self.assertEqual(plan_meta.route, "METADATA_LOOKUP")

        plan_sec = router.route_question("What did the Court order regarding security for costs?")
        self.assertEqual(plan_sec.route, "SECTION_SEARCH")

    def test_pipeline_answer_synthesis(self):
        if not self.pipeline:
            self.skipTest("Preprocessed test docs missing")
        
        # Test Deterministic Bypass
        res_bypass = self.pipeline.answer_question("What is the claim number?")
        self.assertEqual(res_bypass["metadata"]["route"], "deterministic_bypass")
        self.assertIn("CA 005/2025", res_bypass["answer"])

        # Test Hybrid Extractive RAG
        res_rag = self.pipeline.answer_question("What did the Court order regarding the AED 550,000 security for costs?")
        self.assertIn("AED 550,000", res_rag["answer"])
        self.assertGreaterEqual(len(res_rag["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
