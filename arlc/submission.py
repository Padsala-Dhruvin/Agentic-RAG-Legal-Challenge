"""
arlc/submission.py

Submission builder and formatting utilities for the Agentic RAG Legal Challenge.
Formats answers and telemetry into the exact JSON schema required by the evaluation platform.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from arlc.telemetry import Telemetry, normalize_retrieved_pages


@dataclass
class SubmissionAnswer:
    """Represents a single formatted answer and its telemetry payload."""
    question_id: str
    answer: Any
    telemetry: Optional[Telemetry] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to the exact dictionary schema expected by the platform evaluation engine."""
        payload: Dict[str, Any] = {
            "question_id": self.question_id,
            "answer": self.answer,
        }

        if self.telemetry is not None:
            # Normalize retrieval references to deduplicate and sort 1-based physical page numbers
            normalized_refs = normalize_retrieved_pages(self.telemetry.retrieval)
            
            payload["telemetry"] = {
                "timing": {
                    "ttft_ms": self.telemetry.timing.ttft_ms,
                    "tpot_ms": self.telemetry.timing.tpot_ms,
                    "total_time_ms": self.telemetry.timing.total_time_ms,
                },
                "retrieval": {
                    "retrieved_chunk_pages": [
                        {
                            "doc_id": ref.doc_id,
                            "page_numbers": ref.page_numbers,
                        }
                        for ref in normalized_refs
                    ]
                },
                "usage": {
                    "input_tokens": self.telemetry.usage.input_tokens,
                    "output_tokens": self.telemetry.usage.output_tokens,
                },
                "model_name": self.telemetry.model_name,
            }

        return payload


class SubmissionBuilder:
    """
    Stateful builder that accumulates evaluation answers and generates the final
    submission.json file required by the competition platform.
    """

    def __init__(self, architecture_summary: str = "Legal Hybrid RAG (Dense + BM25 RRF + Reranker + Deterministic Bypass)") -> None:
        self.architecture_summary: str = architecture_summary
        self.answers: List[SubmissionAnswer] = []

    def add_answer(self, question_id: str, answer: Any, telemetry: Optional[Telemetry] = None) -> None:
        """Add a completed question answer and its telemetry to the submission payload."""
        self.answers.append(
            SubmissionAnswer(
                question_id=question_id,
                answer=answer,
                telemetry=telemetry,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        """Generate the complete top-level submission dictionary."""
        return {
            "architecture_summary": self.architecture_summary,
            "answers": [ans.to_dict() for ans in self.answers],
        }

    def save(self, output_path: Union[str, Path]) -> Path:
        """Serialize the submission to a JSON file with proper UTF-8 encoding and indentation."""
        file_path = Path(output_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            
        return file_path
