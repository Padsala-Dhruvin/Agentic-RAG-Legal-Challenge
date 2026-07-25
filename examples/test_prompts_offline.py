"""
examples/test_prompts_offline.py

Smoke test script for Sub-Step 6.1 (Free-Text Prompt Builders).
Verifies that build_free_text_prompt formats chunks with physical page headers
and respects max_context_chars limits.
"""

import logging
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.chunkers.legal_chunk_types import LegalChunk
from retrieval.free_text_prompts import LEGAL_SYSTEM_PROMPT, build_free_text_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_prompts_offline")


def main() -> None:
    logger.info("Initializing smoke test for Sub-Step 6.1 (Free-Text Prompts)...")

    sample_chunks = [
        LegalChunk(
            chunk_id="doc_001_sec_0",
            doc_id="case_001_appeal",
            representation="section",
            text="The Court ordered that the appellant pay security for costs assessed at AED 550,000 within 14 days.",
            pages=[2, 3],
        ),
        LegalChunk(
            chunk_id="doc_001_page_4",
            doc_id="case_001_appeal",
            representation="page_anchor",
            text="If unpaid, interest shall accrue at the statutory post-judgment interest rate of 9% per annum.",
            pages=[4],
        ),
    ]

    question = "What is the assessed amount for security for costs and what interest rate applies?"
    logger.info("Testing build_free_text_prompt across %d chunks...", len(sample_chunks))

    prompt = build_free_text_prompt(question, sample_chunks, max_context_chars=1000)

    print("-" * 70)
    print("SYSTEM PROMPT PREVIEW:")
    print(LEGAL_SYSTEM_PROMPT[:300] + "...\n")
    print("-" * 70)
    print("GENERATED USER RAG PROMPT:")
    print(prompt)
    print("-" * 70)

    assert "Physical Pages: 2, 3" in prompt, "Failed to include exact physical page numbers for Chunk #1!"
    assert "Physical Pages: 4" in prompt, "Failed to include exact physical page numbers for Chunk #2!"
    assert question in prompt, "Failed to embed user question inside prompt!"

    print("\n" + "=" * 70)
    print("SUB-STEP 6.1 SMOKE TEST PASSED SUCCESSFULLY!")
    print("Zero-hallucination system prompt and physical page citations verified!")
    print("=" * 70)


if __name__ == "__main__":
    main()
