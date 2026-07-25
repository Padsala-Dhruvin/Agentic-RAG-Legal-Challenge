"""
retrieval/loaders/__init__.py

Exports the IngestedCorpusLoader and data classes.
"""

from retrieval.loaders.ingested_corpus_loader import (
    IngestedCorpusLoader,
    LoadedDocument,
    TextBlock,
)

__all__ = [
    "IngestedCorpusLoader",
    "LoadedDocument",
    "TextBlock",
]
