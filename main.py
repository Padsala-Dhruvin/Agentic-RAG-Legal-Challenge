"""
main.py

Master CLI Entry Point for the Agentic RAG Legal Challenge (`Phase 8.1 & 8.3`).
Provides unified command-line flags to ingest documents, evaluate benchmarks,
and package the final repository into `code_archive.zip` for competition submission.
"""

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arlc.config import get_config
from examples.legal_hybrid_rag import main as eval_main
from ingestion.ingest import Ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("main_cli")


def run_ingestion() -> None:
    """Run offline document ingestion (`Phase 2`)."""
    cfg = get_config()
    logger.info("Starting offline ingestion across corpus path: %s", cfg.docs_dir)
    engine = Ingestion(docs_dir=cfg.docs_dir, output_dir=Path("ingestion/test_ingest_output"), config=cfg)
    results = engine.ingest()
    logger.info("Ingestion complete! Processed %d documents.", results.get("processed", 0))


def run_evaluation() -> None:
    """Run master RAG benchmark evaluation (`Phase 7.2`)."""
    logger.info("Triggering Master RAG Evaluation Runner...")
    eval_main()


def package_code_archive(output_path: str = "code_archive.zip") -> Path:
    """
    Package the codebase into a zip file (`code_archive.zip`) for competition submission (`Phase 8.3`).
    Includes essential packages while excluding virtual environments and cache directories.
    """
    logger.info("Building production code archive: %s...", output_path)
    out_file = Path(output_path)
    if out_file.exists():
        out_file.unlink()

    include_dirs = ["arlc", "ingestion", "retrieval", "examples", "tests"]
    include_files = ["main.py", "update_pipeline.py", "pyproject.toml", "requirements.txt", "README.md", "submission.json", "evaluation_report.json"]

    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add top-level files
        for f in include_files:
            p = Path(f)
            if p.exists():
                zf.write(p, p.name)
                logger.debug("Added file: %s", p.name)

        # Add directories recursively
        for d in include_dirs:
            dir_path = Path(d)
            if not dir_path.exists():
                continue
            for root, dirs, files in os.walk(dir_path):
                # Filter out unwanted subdirectories
                dirs[:] = [sub for sub in dirs if sub not in ("__pycache__", ".pytest_cache", "test_ingest_output")]
                for file in files:
                    if file.endswith(".pyc") or file.endswith(".DS_Store"):
                        continue
                    full_path = Path(root) / file
                    rel_path = full_path.relative_to(Path("."))
                    zf.write(full_path, rel_path)
                    logger.debug("Added file: %s", rel_path)

    logger.info("Successfully created code archive: %s (Size: %.2f KB)", out_file, out_file.stat().st_size / 1024)
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG Legal Challenge — Master CLI Runner")
    parser.add_argument("--ingest", action="store_true", help="Run offline document ingestion over docs_corpus/")
    parser.add_argument("--eval", action="store_true", help="Run benchmark evaluation across questions.json and export submission.json")
    parser.add_argument("--package", action="store_true", help="Package source code into code_archive.zip for platform submission")
    parser.add_argument("--all", action="store_true", help="Run ingestion, evaluation, and packaging sequentially")

    args = parser.parse_args()

    if not any([args.ingest, args.eval, args.package, args.all]):
        parser.print_help()
        print("\nTip: Run `python main.py --eval` to benchmark or `python main.py --package` to build code_archive.zip.")
        sys.exit(0)

    if args.all or args.ingest:
        run_ingestion()
    if args.all or args.eval:
        run_evaluation()
    if args.all or args.package:
        package_code_archive()


if __name__ == "__main__":
    main()
