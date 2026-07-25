"""
retrieval/free_text_prompts.py

Strictly grounded free-text prompt builders for Legal RAG synthesis (Sub-Step 6.1).
Enforces zero-hallucination policies and mandatory physical page citations.
"""

from typing import List
from retrieval.chunkers.legal_chunk_types import LegalChunk

LEGAL_SYSTEM_PROMPT = """You are an expert legal AI assistant designed for high-precision statutory and case law analysis across DIFC, ADGM, and English common law jurisdictions.
Your primary duty is strictly grounded question answering based ONLY on the provided legal text chunks.

MANDATORY RULES:
1. ZERO HALLUCINATION: You must ONLY use facts, dates, monetary amounts, and legal principles explicitly written in the Context Chunks below. Do NOT assume, extrapolate, or use outside legal knowledge.
2. CITATION REQUIREMENT: For every factual assertion you make, you must cite the exact Document ID and Physical Page number from which that fact was drawn using the format: `[Doc: <doc_id>, Page: <page_number>]`.
3. INSUFFICIENT EVIDENCE: If the provided Context Chunks do not contain sufficient information to answer the question with 100% factual certainty, you must reply explicitly: "Insufficient evidence in retrieved documents to answer precisely."
4. CONCISENESS & CLARITY: State the direct answer clearly, followed by the supporting reasoning drawn from the cited chunks."""


def build_free_text_prompt(question: str, chunks: List[LegalChunk], max_context_chars: int = 12000) -> str:
    """Build a structured prompt combining the question and retrieved chunks with page citations."""
    if not chunks:
        context_str = "No relevant legal context chunks retrieved."
    else:
        formatted_chunks = []
        current_chars = 0
        for idx, chunk in enumerate(chunks, 1):
            if current_chars + len(chunk.text) > max_context_chars and idx > 1:
                break
            pages_str = ", ".join(map(str, chunk.pages)) if chunk.pages else "1"
            chunk_header = f"--- [Context Chunk #{idx} | Doc ID: {chunk.doc_id} | Physical Pages: {pages_str} | Type: {chunk.representation}] ---"
            formatted_chunks.append(f"{chunk_header}\n{chunk.text.strip()}\n")
            current_chars += len(chunk.text)
        context_str = "\n".join(formatted_chunks)

    prompt = f"""### RETRIEVED LEGAL CONTEXT:
{context_str}

### USER QUESTION:
{question}

### INSTRUCTIONS:
Answer the question accurately based strictly on the RETRIEVED LEGAL CONTEXT above. Ensure every factual claim includes a page citation like `[Doc: <doc_id>, Page: <page_num>]`."""
    return prompt
