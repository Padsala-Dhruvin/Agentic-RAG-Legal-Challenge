"""
ingestion/__init__.py

Package for offline processing, document structure recognition, and legal metadata extraction.
"""

from ingestion.utils import call_llm, get_llm_client_and_model
from ingestion.legal_metadata import (
    CASE_METADATA_PROMPT,
    LAW_METADATA_PROMPT,
    build_metadata_prompt,
)
from ingestion.ingest import Ingestion

__all__ = [
    "call_llm",
    "get_llm_client_and_model",
    "CASE_METADATA_PROMPT",
    "LAW_METADATA_PROMPT",
    "build_metadata_prompt",
    "Ingestion",
]

