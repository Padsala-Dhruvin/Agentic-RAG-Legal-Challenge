"""
ingestion/ingest.py

Offline Legal Ingestion Engine Orchestrator.
Parses raw legal PDF files into structured text blocks (_content_list.json),
performs heading hierarchy classification (_structure.json), and extracts
schema-validated legal metadata (_metadata.json).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pypdf
from tqdm import tqdm

from arlc.config import EnvConfig, get_config
from ingestion.legal_metadata import build_metadata_prompt
from ingestion.utils import call_llm

logger = logging.getLogger(__name__)


class Ingestion:
    """Orchestrates offline 3-stage ingestion for a corpus of legal PDFs."""

    def __init__(
        self,
        docs_dir: Optional[Union[str, Path]] = None,
        output_dir: Union[str, Path] = "ingestion/docs_corpus_ingest_result",
        config: Optional[EnvConfig] = None,
    ) -> None:
        self.config: EnvConfig = config or get_config()
        self.docs_dir: Path = Path(docs_dir or self.config.docs_dir)
        self.output_dir: Path = Path(output_dir)

    def parse(self, pdf_path: Path, doc_id: str) -> List[Dict[str, Any]]:
        """Stage 1: Extract text blocks per page along with 0-based page_idx."""
        logger.info("[%s] Stage 1: Parsing raw PDF pages...", doc_id)
        reader = pypdf.PdfReader(pdf_path)
        content_list = []
        block_counter = 0

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            for para in paragraphs:
                content_list.append({
                    "block_idx": block_counter,
                    "page_idx": page_idx,
                    "type": "table" if "|" in para and "\n" in para else "text",
                    "text": para,
                })
                block_counter += 1

        # Save _content_list.json
        doc_txt_dir = self.output_dir / doc_id / "txt"
        doc_txt_dir.mkdir(parents=True, exist_ok=True)
        out_path = doc_txt_dir / f"{doc_id}_content_list.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(content_list, f, indent=2, ensure_ascii=False)

        return content_list

    def structure_analysis(self, doc_id: str, content_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Stage 2: Assign heading structure levels (1 to 5) to each block."""
        logger.info("[%s] Stage 2: Performing heading structure analysis...", doc_id)
        doc_txt_dir = self.output_dir / doc_id / "txt"
        out_path = doc_txt_dir / f"{doc_id}_structure.json"

        # If LLM ingestion is disabled, apply fast heuristic fallbacks
        if not self.config.ingest_use_llm:
            structure = []
            for item in content_list:
                text = item["text"]
                level = 1 if len(text) < 60 and text.isupper() else (2 if text.startswith(("Part ", "Chapter ", "Article ")) else 4)
                structure.append({"block_idx": item["block_idx"], "level": level})
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(structure, f, indent=2)
            return structure

        # Construct /no_think prompt for heading hierarchy classification
        sample_blocks = "\n".join([f"[{item['block_idx']}] {item['text'][:150]}" for item in content_list[:100]])
        prompt = (
            "/no_think\n"
            "You are an expert document layout classifier. Assign a structural hierarchy level (1 to 5) to each block.\n"
            "Level 1: Document Title / Major Header\n"
            "Level 2: Part / Chapter Header\n"
            "Level 3: Section / Article Header\n"
            "Level 4: Standard Paragraph / Body Text\n"
            "Level 5: Footnote / Page Number\n\n"
            "Return ONLY a JSON object with format: {\"structure\": [{\"block_idx\": 0, \"level\": 1}, ...]}\n\n"
            f"Blocks:\n{sample_blocks}"
        )

        try:
            raw_response = call_llm(prompt, self.config, json_mode=True)
            payload = json.loads(raw_response)
            structure = payload.get("structure", [])
            # Fallback for unclassified blocks
            existing_indices = {s["block_idx"] for s in structure if "block_idx" in s}
            for item in content_list:
                if item["block_idx"] not in existing_indices:
                    structure.append({"block_idx": item["block_idx"], "level": 4})
        except Exception as exc:
            logger.warning("[%s] Structure classification failed (%s). Using fallback levels.", doc_id, exc)
            structure = [{"block_idx": item["block_idx"], "level": 4} for item in content_list]

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(structure, f, indent=2)
        return structure

    def metadata_extraction(self, doc_id: str, content_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 3: Extract structured legal facts and article page indices."""
        logger.info("[%s] Stage 3: Extracting legal metadata and indices...", doc_id)
        doc_txt_dir = self.output_dir / doc_id / "txt"
        out_path = doc_txt_dir / f"{doc_id}_metadata.json"

        # Determine if case or law based on heuristics
        sample_text = "\n\n".join([item["text"] for item in content_list[:15]])
        is_law = any(k in sample_text.lower() for k in ["law no.", "statute", "enacted by", "part 1 - general"]) and " v " not in sample_text.lower()
        doc_type = "law" if is_law else "case"

        if not self.config.ingest_use_llm:
            fallback_meta = {
                "doc_type": doc_type,
                "doc_id": doc_id,
                "claim_number": doc_id if doc_type == "case" else None,
                "official_title": doc_id.replace("_", " ") if doc_type == "law" else None,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(fallback_meta, f, indent=2)
            return fallback_meta

        prompt = build_metadata_prompt(doc_type, sample_text)
        try:
            raw_response = call_llm(prompt, self.config, json_mode=True)
            metadata = json.loads(raw_response)
        except Exception as exc:
            logger.warning("[%s] Metadata extraction failed (%s). Using default metadata.", doc_id, exc)
            metadata = {"doc_type": doc_type, "doc_id": doc_id}

        metadata["doc_id"] = doc_id
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return metadata

    def ingest(self) -> Dict[str, Any]:
        """Execute the 3-stage ingestion pipeline across all PDF files in docs_dir."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {self.docs_dir}")

        pdf_files = list(self.docs_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s to ingest.", self.docs_dir)
            return {"processed": 0, "documents": []}

        logger.info("Starting offline ingestion for %d PDF documents in %s...", len(pdf_files), self.docs_dir)
        processed_docs = []

        for pdf_path in tqdm(pdf_files, desc="Ingesting Legal Corpus"):
            doc_id = pdf_path.stem
            content_list = self.parse(pdf_path, doc_id)
            self.structure_analysis(doc_id, content_list)
            self.metadata_extraction(doc_id, content_list)
            processed_docs.append(doc_id)

        return {"processed": len(processed_docs), "documents": processed_docs}
