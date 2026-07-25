"""
retrieval/legal_question_router.py

Legal Question Router (Phase 4).
Classifies incoming questions using regex heuristics into specialized routes:
- METADATA_LOOKUP: Fast-path answering from cover page / structured facts.
- SECTION_SEARCH: Hybrid RRF search across section & page anchor chunks.
- GLOBAL_SYNTHESIS: Multi-document summary or broad cross-case analysis.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class RoutePlan:
    """Represents the routing decision for an incoming legal question (Sub-Step 4.1)."""
    route: str  # 'METADATA_LOOKUP', 'SECTION_SEARCH', or 'GLOBAL_SYNTHESIS'
    target_doc_id: Optional[str] = None
    confidence: float = 1.0
    extracted_filters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


class LegalQuestionRouter:
    """Regex-based detection engine for classifying legal queries (Sub-Step 4.2)."""

    def __init__(self) -> None:
        # Regex patterns for metadata fact queries
        self.metadata_patterns = [
            r"\b(who|name of)\s+(is|was|are|were)\s+the\s+(judge|judges|justice|claimant|defendant|appellant|respondent)\b",
            r"\bwhat\s+is\s+the\s+(claim\s+number|case\s+name|neutral\s+citation|date|issue\s+date|judgment\s+date)\b",
            r"\bwhen\s+was\s+(the\s+)?(judgment|order|claim)\s+(issued|dated|decided)\b",
        ]

        # Regex patterns for extracting DIFC claim numbers (e.g., CA 005/2025 or CFI-073-2024)
        self.claim_num_pattern = re.compile(r"\b([A-Z]{2,3}[\s\-]\d{3}[\/\-]\d{4})\b", re.IGNORECASE)
        
        # Regex pattern for 64-character hex document IDs
        self.doc_id_pattern = re.compile(r"\b([a-fA-F0-9]{64})\b")

    def route_question(self, query: str) -> RoutePlan:
        """Analyze query string and return a RoutePlan."""
        q_lower = query.lower()
        extracted_filters: Dict[str, Any] = {}
        target_doc_id: Optional[str] = None

        # 1. Extract Document ID or Claim Number citation if present
        doc_id_match = self.doc_id_pattern.search(query)
        if doc_id_match:
            target_doc_id = doc_id_match.group(1).lower()
            extracted_filters["doc_id"] = target_doc_id

        claim_match = self.claim_num_pattern.search(query)
        if claim_match:
            claim_num = claim_match.group(1).replace("-", " ").replace("/", " ").upper()
            extracted_filters["claim_number"] = claim_match.group(1)
            # If we don't have a direct hex ID, we record the claim number as our filter anchor
            if not target_doc_id:
                extracted_filters["target_claim_number"] = claim_num

        # 2. Check if question asks for global summaries or multi-case comparisons
        if any(w in q_lower for w in ("summarize all", "compare all", "across all cases", "list all judgments")):
            return RoutePlan(
                route="GLOBAL_SYNTHESIS",
                target_doc_id=target_doc_id,
                confidence=0.9,
                extracted_filters=extracted_filters,
                reasoning="Detected multi-case synthesis keywords."
            )

        # 3. Check if question is a fast-path cover page / fact lookup
        for pat in self.metadata_patterns:
            if re.search(pat, q_lower):
                return RoutePlan(
                    route="METADATA_LOOKUP",
                    target_doc_id=target_doc_id,
                    confidence=0.95,
                    extracted_filters=extracted_filters,
                    reasoning=f"Matched metadata lookup pattern: {pat}"
                )

        # 4. Default to Hybrid Section Search for substantive legal issues, holdings, and rules
        return RoutePlan(
            route="SECTION_SEARCH",
            target_doc_id=target_doc_id,
            confidence=0.85,
            extracted_filters=extracted_filters,
            reasoning="Defaulted to hybrid section & page anchor search for substantive legal query."
        )
