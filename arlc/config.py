"""
arlc/config.py

Environment Configuration loader for the Agentic RAG Legal Challenge.
Exposes runtime settings via a strongly-typed EnvConfig dataclass.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


@dataclass
class EnvConfig:
    """Runtime configuration loaded from environment variables (.env)."""
    # Evaluation API settings (Optional if running against remote REST API)
    eval_api_key: Optional[str]
    eval_base_url: Optional[str]

    # LLM configuration
    openrouter_api_key: Optional[str]
    openai_api_key: Optional[str]
    gemini_api_key: Optional[str]
    llm_model: str

    # Embeddings configuration
    use_cohere_embeddings: bool
    use_openai_embeddings: bool
    cohere_api_key: Optional[str]
    embedding_model: str

    # Optional API Rerankers
    voyage_api_key: Optional[str]

    # Remote Vector Store (Turbopuffer)
    turbopuffer_api_key: Optional[str]
    turbopuffer_namespace: str
    skip_indexing: bool

    # Pipeline behavior flags
    enable_rerank: bool
    mock_llm: bool
    ingest_use_llm: bool

    # File paths and Local Dataset configuration
    docs_dir: str
    questions_path: str
    submission_path: str
    output_report_path: str
    code_archive_path: str

    @classmethod
    def from_env(cls) -> "EnvConfig":
        """Load settings from .env file or OS environment variables."""
        load_dotenv()

        def _as_bool(val: str, default: bool) -> bool:
            if val is None:
                return default
            return str(val).strip().lower() in ("1", "true", "yes", "y")

        return cls(
            eval_api_key=os.getenv("EVAL_API_KEY") or None,
            eval_base_url=os.getenv("EVAL_BASE_URL") or None,
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip() or None,
            openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
            gemini_api_key=(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip() or None,
            llm_model=os.getenv("LLM_MODEL", "gemini-flash-latest").strip(),
            use_cohere_embeddings=_as_bool(os.getenv("USE_COHERE_EMBEDDINGS"), True),
            use_openai_embeddings=_as_bool(os.getenv("USE_OPENAI_EMBEDDINGS"), False),
            cohere_api_key=os.getenv("COHERE_API_KEY") or None,
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            voyage_api_key=os.getenv("VOYAGE_API_KEY") or None,
            turbopuffer_api_key=os.getenv("TURBOPUFFER_API_KEY") or None,
            turbopuffer_namespace=os.getenv("TURBOPUFFER_NAMESPACE", "legal-challenge-final"),
            skip_indexing=_as_bool(os.getenv("LEGAL_HYBRID_SKIP_INDEXING"), False),
            enable_rerank=_as_bool(os.getenv("LEGAL_HYBRID_ENABLE_RERANK"), True),
            mock_llm=_as_bool(os.getenv("LEGAL_RAG_SMOKE_MOCK_LLM"), False),
            ingest_use_llm=_as_bool(os.getenv("LEGAL_INGEST_USE_LLM"), True),
            docs_dir=os.getenv("DOCS_DIR", "docs_corpus"),
            questions_path=os.getenv("QUESTIONS_PATH", "public_dataset/questions.json"),
            submission_path=os.getenv("SUBMISSION_PATH", "submission.json"),
            output_report_path=os.getenv("OUTPUT_REPORT_PATH", "evaluation_report.json"),
            code_archive_path=os.getenv("CODE_ARCHIVE_PATH", "code_archive.zip"),
        )


def get_config() -> EnvConfig:
    """Instantiate and return the global runtime EnvConfig."""
    return EnvConfig.from_env()
