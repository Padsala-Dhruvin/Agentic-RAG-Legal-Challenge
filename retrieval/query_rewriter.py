"""
retrieval/query_rewriter.py

Conservative query rewriting for legal retrieval.
The rewriter expands vague or colloquial user queries without changing
deterministic metadata intent.
"""

import re
from typing import List


class LegalQueryRewriter:
    """Generate safe retrieval query variants for SECTION_SEARCH."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._phrase_map = {
            "high profile": "significant",
            "top case": "significant judgment",
            "most important": "significant judgment",
            "give me": "find",
            "what happened": "final order and holding",
            "tell me": "find",
            "about": "regarding",
            "all judge name": "list presiding judge names",
            "judge name": "presiding judge name",
            "file case": "case",
        }
        self._legal_keywords = {
            "case",
            "claim",
            "judgment",
            "order",
            "article",
            "law",
            "court",
            "appeal",
            "liability",
            "damages",
        }

    def rewrite(self, query: str) -> str:
        """Return a conservative rewritten query for legal retrieval."""
        cleaned = " ".join((query or "").strip().split())
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        for old, new in self._phrase_map.items():
            lowered = lowered.replace(old, new)

        # Preserve legal identifiers while normalizing punctuation/spacing.
        lowered = re.sub(r"\s+", " ", lowered).strip(" .,!?:;")

        tokens = set(re.findall(r"\w+", lowered))
        if not (tokens & self._legal_keywords):
            lowered = f"{lowered} legal case judgment"

        return lowered

    def generate_candidates(self, query: str, route: str) -> List[str]:
        """Return ordered unique candidate queries for retrieval."""
        base = " ".join((query or "").strip().split())
        if not base:
            return []
        if not self.enabled or route != "SECTION_SEARCH":
            return [base]

        candidates: List[str] = [base]
        rewritten = self.rewrite(base)
        if rewritten and rewritten.lower() != base.lower():
            candidates.append(rewritten)

        lowered = base.lower()
        # Intent-specific focused rewrites for common vague prompts.
        if re.search(r"\b(high\s*profile|significant|important)\b", lowered):
            candidates.append("most significant DIFC court judgment with key legal holding")
        if re.search(r"\b(list|all)\b.*\bjudge", lowered):
            candidates.append("list presiding judge names from title page and judgment headers")
        if re.search(r"\bwhich court\b|\bcourt\b.*\bcase", lowered):
            candidates.append("court name claim number and case metadata")

        claim_match = re.search(r"\b([A-Z]{2,3}[\s\-]?\d+[\/\-]\d{4})\b", base, re.IGNORECASE)
        article_match = re.search(r"\barticle\s+\d+(?:\(\d+\))?\b", base, re.IGNORECASE)
        if claim_match:
            candidates.append(f"case claim number {claim_match.group(1)} judgment order")
        if article_match:
            candidates.append(f"{article_match.group(0)} legal interpretation and application")

        unique: List[str] = []
        seen = set()
        for q in candidates:
            key = q.lower().strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(q)
        return unique[:2]
