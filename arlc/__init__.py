"""
arlc/__init__.py

Core library package for the Legal Hybrid RAG benchmarking and evaluation suite.
Re-exports key configuration, telemetry, submission builder, and dataset client classes.
"""

from arlc.config import EnvConfig, get_config
from arlc.telemetry import (
    RetrievalRef,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    normalize_retrieved_pages,
)
from arlc.submission import SubmissionAnswer, SubmissionBuilder
from arlc.client import EvaluationClient

__all__ = [
    "EnvConfig",
    "get_config",
    "RetrievalRef",
    "Telemetry",
    "TelemetryTimer",
    "TimingMetrics",
    "UsageMetrics",
    "normalize_retrieved_pages",
    "SubmissionAnswer",
    "SubmissionBuilder",
    "EvaluationClient",
]
