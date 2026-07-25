"""
ingestion/legal_metadata.py

Structured prompt templates and builders for extracting legal metadata during offline ingestion.
Produces schema-compliant metadata dicts for both court cases and statutory laws.
"""

from typing import Dict, Any

CASE_METADATA_PROMPT = """
You are an expert legal metadata extraction engine specializing in court case judgments.
Analyze the provided text from a legal case document (especially the title/cover pages) and extract exact factual fields.

Return ONLY a valid JSON object matching this schema:
{
  "doc_type": "case",
  "claim_number": "Exact claim/case number if present (e.g. CA 005/2025 or CFI 057/2025), else null",
  "claim_number_normalized": "Lowercase alphanumeric representation without punctuation (e.g. ca 005 2025), else null",
  "neutral_citation": "Official neutral citation if present (e.g. [2025] DIFC CA 005), else null",
  "neutral_citation_normalized": "Lowercase alphanumeric neutral citation (e.g. 2025 difc ca 005), else null",
  "case_name": "Name of the case (e.g. Smith v Jones), else null",
  "claimant": "Name of claimant/appellant/applicant if explicit, else null",
  "defendants": ["List of defendant/respondent names if explicit, else []"],
  "issue_date": "Date of issue in YYYY-MM-DD format if explicit, else null",
  "judgment_date": "Date of judgment/order in YYYY-MM-DD format if explicit, else null",
  "judge": "Name of presiding judge(s) or justice(s) if explicit, else null"
}

Rules:
- If a field is not clearly stated in the text, return null (or empty list for defendants).
- Do not guess or hallucinate names or dates.
- For dates, convert natural dates (e.g. '15 March 2025') to strict ISO format ('2025-03-15').
"""

LAW_METADATA_PROMPT = """
You are an expert legal metadata extraction engine specializing in legislative statutory laws and codes.
Analyze the provided text from a law document and extract exact factual fields along with article locations.

Return ONLY a valid JSON object matching this schema:
{
  "doc_type": "law",
  "official_title": "Full official title of the law (e.g. DIFC Employment Law 2024), else null",
  "short_title": "Common or short title (e.g. Employment Law), else null",
  "law_number": 2,
  "official_citation": "Official statutory citation (e.g. DIFC Law No. 2 of 2024), else null",
  "official_citation_normalized": "Lowercase alphanumeric citation (e.g. difc law no 2 of 2024), else null",
  "alias_keys": ["List of common lowercase search phrases for this law, e.g. 'employment law', 'difc employment law 2024'"],
  "article_index": {
    "Article 1": [1, 2],
    "Article 2": [2, 3]
  },
  "section_heading_index": {
    "Part 1 - General": [1],
    "Definitions": [3]
  }
}

Rules:
- Extract all prominent Article and Part headers mentioned in the sample and map them to their physical page numbers.
- If a field is not present, return null.
"""


def build_metadata_prompt(doc_type: str, document_sample: str) -> str:
    """
    Construct the final prompt string to send to the LLM for structured metadata extraction.
    """
    base_prompt = CASE_METADATA_PROMPT if doc_type.lower() == "case" else LAW_METADATA_PROMPT
    return f"{base_prompt.strip()}\n\n--- DOCUMENT SAMPLE ---\n{document_sample.strip()}"
