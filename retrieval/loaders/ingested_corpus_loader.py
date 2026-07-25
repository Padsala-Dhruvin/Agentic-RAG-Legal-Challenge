"""
retrieval/loaders/ingested_corpus_loader.py

Loads preprocessed JSON outputs (_content_list.json, _structure.json, _metadata.json)
from the offline ingestion directory and reconstructs unified Document objects
for downstream chunking and vector indexing.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a single paragraph or table block from _content_list.json."""
    block_idx: int
    page_idx: int  # 0-based page number from pypdf
    block_type: str  # 'text' or 'table'
    text: str
    level: int = 4  # Hierarchy level from _structure.json (1=title, 2=chapter, 3=article, 4=body)


@dataclass
class LoadedDocument:
    """Represents a complete legal document with merged content blocks and metadata."""
    doc_id: str
    metadata: Dict[str, Any]
    blocks: List[TextBlock] = field(default_factory=list)

    @property
    def total_pages(self) -> int:
        """Return total number of physical pages in the document."""
        if not self.blocks:
            return 0
        return max(b.page_idx for b in self.blocks) + 1


class IngestedCorpusLoader:
    """Loads and merges offline ingestion artifacts from disk."""

    def __init__(self, ingest_dir: Union[str, Path] = "ingestion/docs_corpus_ingest_result") -> None:
        self.ingest_dir: Path = Path(ingest_dir)

    def load_document(self, doc_id: str) -> Optional[LoadedDocument]:
        """Load _content_list, _structure, and _metadata for a single doc_id and merge them."""
        doc_txt_dir = self.ingest_dir / doc_id / "txt"
        content_path = doc_txt_dir / f"{doc_id}_content_list.json"
        structure_path = doc_txt_dir / f"{doc_id}_structure.json"
        metadata_path = doc_txt_dir / f"{doc_id}_metadata.json"

        if not content_path.exists():
            logger.warning("Missing _content_list.json for %s at %s", doc_id, content_path)
            return None

        with open(content_path, "r", encoding="utf-8") as f:
            raw_contents = json.load(f)

        structure_map = {}
        if structure_path.exists():
            with open(structure_path, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if "block_idx" in item and "level" in item:
                        structure_map[item["block_idx"]] = item["level"]

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        blocks = []
        for item in raw_contents:
            b_idx = item.get("block_idx", len(blocks))
            blocks.append(TextBlock(
                block_idx=b_idx,
                page_idx=item.get("page_idx", 0),
                block_type=item.get("type", "text"),
                text=item.get("text", ""),
                level=structure_map.get(b_idx, 4),
            ))

        return LoadedDocument(doc_id=doc_id, metadata=metadata, blocks=blocks)

    def load_all_documents(self) -> List[LoadedDocument]:
        """Scan ingest_dir and load all preprocessed legal documents."""
        if not self.ingest_dir.exists():
            logger.error("Ingest directory does not exist: %s", self.ingest_dir)
            return []

        loaded_docs = []
        for doc_folder in sorted(self.ingest_dir.iterdir()):
            if doc_folder.is_dir():
                doc_id = doc_folder.name
                doc = self.load_document(doc_id)
                if doc:
                    loaded_docs.append(doc)

        logger.info("Successfully loaded %d preprocessed legal documents from %s", len(loaded_docs), self.ingest_dir)
        return loaded_docs
