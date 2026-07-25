"""
update_pipeline.py

Automated Pipeline Updater & Synchronization Utility (`Sub-Step 8.2`).
Detects new legal PDFs dropped into docs_corpus/, runs offline ingestion on missing files,
and refreshes the hybrid search index (and optional cloud vector store).
"""

import logging
import sys
from pathlib import Path
from typing import List, Set

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arlc.config import get_config
from ingestion.ingest import Ingestion
from retrieval import IngestedCorpusLoader, LegalHybridRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("update_pipeline")


def detect_new_documents(docs_dir: Path, ingest_dir: Path) -> List[Path]:
    """Identify PDF files in docs_dir that do not have existing extraction results in ingest_dir."""
    if not docs_dir.exists():
        return []
    
    existing_ids: Set[str] = set()
    if ingest_dir.exists():
        existing_ids = {d.name for d in ingest_dir.iterdir() if d.is_dir()}

    new_pdfs = [pdf for pdf in docs_dir.glob("*.pdf") if pdf.stem not in existing_ids]
    return new_pdfs


def sync_pipeline() -> LegalHybridRAGPipeline:
    """Detect changes, run ingestion if needed, and re-index the RAG pipeline."""
    cfg = get_config()
    docs_dir = Path(cfg.docs_dir)
    ingest_dir = Path("ingestion/test_ingest_output")

    logger.info("Checking for new PDF documents across corpus: %s", docs_dir)
    new_pdfs = detect_new_documents(docs_dir, ingest_dir)

    if new_pdfs:
        logger.info("Detected %d new/unprocessed PDF documents: %s", len(new_pdfs), [p.name for p in new_pdfs])
        engine = Ingestion(docs_dir=docs_dir, output_dir=ingest_dir, config=cfg)
        engine.ingest()
    else:
        logger.info("All PDF documents in %s are already ingested up-to-date.", docs_dir)

    logger.info("Reloading preprocessed corpus and rebuilding Hybrid RAG Index...")
    loader = IngestedCorpusLoader(ingest_dir=ingest_dir)
    docs = loader.load_all_documents()

    pipeline = LegalHybridRAGPipeline(config=cfg)
    pipeline.index_corpus(docs)

    logger.info("Pipeline synchronization complete! Active index: %d documents, %d chunks.", len(docs), len(pipeline.chunks))
    return pipeline


def main() -> None:
    logger.info("Initializing Legal RAG Pipeline Updater...")
    try:
        sync_pipeline()
        print("\n" + "=" * 70)
        print("PIPELINE UPDATER COMPLETED SUCCESSFULLY! Corpus and indices are synchronized.")
        print("=" * 70)
    except Exception as exc:
        logger.error("Pipeline synchronization failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
