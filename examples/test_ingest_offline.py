"""
examples/test_ingest_offline.py

Smoke test script to verify Phase 2 (Offline Legal Ingestion Engine).
Runs the 3-stage Ingestion pipeline on a sample PDF from public_dataset/docs_corpus/
and checks that _content_list.json, _structure.json, and _metadata.json are correctly created.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure root directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arlc import get_config
from ingestion import Ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_ingest_offline")


def main() -> None:
    logger.info("Initializing smoke test for Phase 2 (Offline Ingestion)...")
    config = get_config()

    # Use public_dataset/docs_corpus as our test directory
    corpus_dir = Path("public_dataset/docs_corpus")
    if not corpus_dir.exists() or not any(corpus_dir.glob("*.pdf")):
        logger.error("No PDF files found in %s. Cannot run test.", corpus_dir)
        sys.exit(1)

    # Pick the first PDF file for a fast single-document verification test
    sample_pdf = next(corpus_dir.glob("*.pdf"))
    doc_id = sample_pdf.stem
    logger.info("Selected sample PDF for testing: %s (%s)", sample_pdf.name, doc_id)

    # Instantiate Ingestion engine with a dedicated test output folder
    test_output_dir = Path("ingestion/test_ingest_output")
    engine = Ingestion(docs_dir=corpus_dir, output_dir=test_output_dir, config=config)

    # Stage 1: Parse
    logger.info("--- Testing Stage 1: parse() ---")
    content_list = engine.parse(sample_pdf, doc_id)
    logger.info("Stage 1 complete! Extracted %d text/table blocks across %d pages.",
                len(content_list), max([item["page_idx"] for item in content_list] + [0]) + 1)

    # Stage 2: Structure Analysis (Using fast heuristic fallback if LLM not explicitly enabled/keyed for test)
    logger.info("--- Testing Stage 2: structure_analysis() ---")
    structure = engine.structure_analysis(doc_id, content_list)
    logger.info("Stage 2 complete! Assigned levels to %d blocks.", len(structure))

    # Stage 3: Metadata Extraction
    logger.info("--- Testing Stage 3: metadata_extraction() ---")
    metadata = engine.metadata_extraction(doc_id, content_list)
    logger.info("Stage 3 complete! Extracted document type: '%s' and metadata keys: %s",
                metadata.get("doc_type"), list(metadata.keys()))

    # Verify files on disk
    doc_txt_dir = test_output_dir / doc_id / "txt"
    content_file = doc_txt_dir / f"{doc_id}_content_list.json"
    structure_file = doc_txt_dir / f"{doc_id}_structure.json"
    metadata_file = doc_txt_dir / f"{doc_id}_metadata.json"

    assert content_file.exists(), f"Missing {content_file}"
    assert structure_file.exists(), f"Missing {structure_file}"
    assert metadata_file.exists(), f"Missing {metadata_file}"

    logger.info("✅ All 3 ingestion output files verified successfully at: %s", doc_txt_dir)
    print("\n" + "=" * 60)
    print("PHASE 2 SMOKE TEST PASSED SUCCESSFULLY!")
    print(f"Sample Metadata Output:\n{json.dumps(metadata, indent=2)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
