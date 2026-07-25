"""
arlc/telemetry.py

Telemetry and Provenance tracking for the Agentic RAG Legal Challenge.
Tracks latency metrics (TTFT, TPOT, Total Time), token usage, and normalizes
physical PDF page citations required for the grounding score multiplier.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class TimingMetrics:
    """Latency metrics in milliseconds."""
    ttft_ms: float
    tpot_ms: float
    total_time_ms: float


@dataclass
class RetrievalRef:
    """Evidence citation pointing to specific 1-based physical PDF page numbers."""
    doc_id: str
    page_numbers: List[int] = field(default_factory=list)


@dataclass
class UsageMetrics:
    """Token consumption metrics reported by the LLM."""
    input_tokens: int
    output_tokens: int


@dataclass
class Telemetry:
    """Complete telemetry record attached to every competition answer."""
    timing: TimingMetrics
    retrieval: List[RetrievalRef]
    usage: UsageMetrics
    model_name: str


def normalize_retrieved_pages(refs: List[RetrievalRef]) -> List[RetrievalRef]:
    """
    Merge duplicate doc_ids, deduplicate 1-based page numbers, and sort them.
    
    Example:
        Input: [
            RetrievalRef("docB", [3, 2]),
            RetrievalRef("docA", [1]),
            RetrievalRef("docB", [2, 4])
        ]
        Output: [
            RetrievalRef("docA", [1]),
            RetrievalRef("docB", [2, 3, 4])
        ]
    """
    doc_pages: Dict[str, Set[int]] = {}

    for ref in refs:
        if not ref.doc_id:
            continue
        if ref.doc_id not in doc_pages:
            doc_pages[ref.doc_id] = set()
        # Ensure only positive 1-based integers are included
        for p in ref.page_numbers:
            if isinstance(p, int) and p > 0:
                doc_pages[ref.doc_id].add(p)

    normalized: List[RetrievalRef] = []
    for doc_id in sorted(doc_pages.keys()):
        pages = sorted(list(doc_pages[doc_id]))
        normalized.append(RetrievalRef(doc_id=doc_id, page_numbers=pages))

    return normalized


class TelemetryTimer:
    """High-precision latency timer tracking TTFT and TPOT."""

    def __init__(self) -> None:
        self._start_time: float = time.perf_counter()
        self._first_token_time: float = 0.0

    def __enter__(self) -> "TelemetryTimer":
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def mark_token(self) -> None:
        """Mark the arrival timestamp of the first LLM token."""
        if self._first_token_time == 0.0:
            self._first_token_time = time.perf_counter()

    def finish(self, output_token_count: int = 1) -> TimingMetrics:
        """Calculate and return TimingMetrics in milliseconds."""
        end_time = time.perf_counter()
        total_time_ms = round((end_time - self._start_time) * 1000.0, 2)

        # If mark_token() was never called (e.g., synchronous LLM call or bypass),
        # TTFT equals the total execution time.
        if self._first_token_time == 0.0:
            ttft_ms = total_time_ms
            tpot_ms = 0.0
        else:
            ttft_ms = round((self._first_token_time - self._start_time) * 1000.0, 2)
            remaining_time_ms = (end_time - self._first_token_time) * 1000.0
            # TPOT measures average latency per subsequent token
            subsequent_tokens = max(1, output_token_count - 1)
            tpot_ms = round(remaining_time_ms / subsequent_tokens, 2)

        return TimingMetrics(
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_time_ms=total_time_ms,
        )
