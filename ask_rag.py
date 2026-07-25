"""
ask_rag.py

Interactive Live Legal Question Answering Script with FULL BEHIND-THE-SCENES VISIBILITY.
Shows you every step of the RAG pipeline as it executes — routing, retrieval, synthesis, and telemetry.
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arlc.config import get_config
from arlc.telemetry import TelemetryTimer
from retrieval import IngestedCorpusLoader, LegalChunker, LegalHybridRAGPipeline, LegalQuestionRouter

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ask_rag")


def main() -> None:
    print("\n" + "=" * 70)
    print("🤖 AGENTIC RAG LEGAL CHALLENGE — LIVE INTERACTIVE DEMO")
    print("=" * 70)
    print("Initializing system and loading legal case documents...")

    cfg = get_config()
    ingest_dir = Path("ingestion/test_ingest_output")
    if not ingest_dir.exists():
        print(f"❌ Error: Preprocessed corpus not found at {ingest_dir}. Run `python main.py --ingest` first.")
        sys.exit(1)

    loader = IngestedCorpusLoader(ingest_dir=ingest_dir)
    docs = loader.load_all_documents()

    pipeline = LegalHybridRAGPipeline(config=cfg)
    pipeline.index_corpus(docs)

    print(f"✅ System Ready! Loaded {len(docs)} legal documents ({len(pipeline.chunks)} searchable chunks).")
    print(f"⚡ Active Synthesis Mode: {pipeline.active_model}")
    print("-" * 70)
    print("Type your legal question below (or 'exit' to quit):\n")

    router = LegalQuestionRouter()

    while True:
        try:
            query = input("📝 Enter Question > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Goodbye! 👋")
                break

            # ============================================
            # STEP 1: ROUTING — What kind of question is this?
            # ============================================
            print("\n" + "=" * 60)
            print("🔍 STEP 1: ROUTING (analyzing your question...)")
            plan = router.route_question(query)
            print(f"   Route classified as: {plan.route}")
            print(f"   Target document ID:  {plan.target_doc_id or 'None (search all docs)'}")
            print(f"   Confidence:          {plan.confidence}")
            print(f"   Reasoning:           {plan.reasoning}")

            # ============================================
            # STEP 2-5: Full pipeline execution with telemetry
            # ============================================
            with TelemetryTimer() as timer:
                res = pipeline.answer_question(query)
                timer.mark_token()
                timing = timer.finish(output_token_count=len(res["answer"].split()))

            route_used = res["metadata"]["route"]
            mode_used = res["metadata"]["mode"]

            if route_used == "deterministic_bypass":
                # ============================================
                # FAST PATH: Deterministic Bypass
                # ============================================
                print("\n⚡ STEP 2: DETERMINISTIC BYPASS (instant metadata lookup)")
                print(f"   This question was answered INSTANTLY from indexed metadata.")
                print(f"   No search needed. No Gemini API call needed. 0 tokens used.")
            else:
                # ============================================
                # FULL RAG PATH: Search → Retrieve → Synthesize
                # ============================================
                chunks_count = res["metadata"].get("chunks_retrieved", 0)
                top_score = res["metadata"].get("top_score", 0.0)

                print(f"\n📊 STEP 2: HYBRID SEARCH (BM25 keyword + TF-IDF vector)")
                print(f"   Searched across {len(pipeline.chunks)} chunks from {len(docs)} documents")
                print(f"   Top {chunks_count} chunks retrieved using Reciprocal Rank Fusion (RRF)")
                print(f"   Best match score: {top_score:.4f}")

                print(f"\n🤖 STEP 3: LLM SYNTHESIS (sending retrieved pages to Gemini)")
                print(f"   Model used: {mode_used}")
                print(f"   Your question + the top {chunks_count} retrieved chunks were bundled")
                print(f"   into a prompt and sent to Google Gemini for answer generation.")

            # ============================================
            # STEP 4: TELEMETRY — How fast was it?
            # ============================================
            print(f"\n⏱️  TELEMETRY:")
            print(f"   Time-To-First-Token (TTFT): {timing.ttft_ms:.2f} ms")
            print(f"   Time-Per-Output-Token (TPOT): {timing.tpot_ms:.2f} ms")
            print(f"   Total Latency:               {timing.total_time_ms:.2f} ms")
            print(f"   Answer Length:                {len(res['answer'].split())} words")

            # ============================================
            # FINAL: The Answer
            # ============================================
            print("\n" + "-" * 60)
            print("💡 ANSWER:")
            print(res["answer"])
            print("\n📌 CITATIONS (Verified 1-Based Physical PDF Pages):")
            for cit in res.get("citations", []):
                print(f"   • Doc: {cit.get('doc_id', '?')[:20]}... | Page: {cit.get('page', 1)}")
            print("=" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
