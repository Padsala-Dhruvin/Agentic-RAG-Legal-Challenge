"""
ingestion/utils.py

Multi-provider LLM helper utility for the offline ingestion engine.
Supports universal routing across Google Gemini Free API, OpenRouter Free (NVIDIA/Llama/Gemini),
and direct OpenAI API using the standard OpenAI REST client abstraction.
"""

import logging
from typing import Optional, Tuple

from openai import OpenAI, OpenAIError

from arlc.config import EnvConfig, get_config

logger = logging.getLogger(__name__)


def get_llm_client_and_model(config: Optional[EnvConfig] = None) -> Tuple[OpenAI, str]:
    """
    Detect the active LLM provider based on EnvConfig keys and model name,
    and return an initialized OpenAI client along with the exact model identifier.
    """
    cfg = config or get_config()
    model = cfg.llm_model.strip()

    # 1. OpenRouter Provider (e.g. nvidia/llama-3.1-nemotron-70b-instruct:free, google/gemini-2.0-flash-lite:free)
    if cfg.openrouter_api_key and ("/" in model or "openrouter" in model.lower()):
        logger.debug("Routing LLM call to OpenRouter API for model: %s", model)
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=cfg.openrouter_api_key.strip(),
        )
        return client, model

    # 2. Google Gemini Free Provider via OpenAI-compatible REST endpoint
    if cfg.gemini_api_key and model.lower().startswith("gemini"):
        logger.debug("Routing LLM call to Google Gemini API for model: %s", model)
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=cfg.gemini_api_key.strip(),
        )
        return client, model

    # 3. Direct OpenAI Provider
    if cfg.openai_api_key:
        logger.debug("Routing LLM call to direct OpenAI API for model: %s", model)
        client = OpenAI(api_key=cfg.openai_api_key.strip())
        return client, model

    raise ValueError(
        f"No valid API key found for model '{model}'. "
        "Please check your .env and set GEMINI_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY."
    )


def call_llm(
    prompt: str,
    config: Optional[EnvConfig] = None,
    system_prompt: str = "You are an expert AI assistant specializing in legal document analysis.",
    json_mode: bool = False,
    temperature: float = 0.0,
) -> str:
    """
    Execute a chat completion request against the configured LLM provider.
    If json_mode is True, instructs the model to return strict JSON formatting.
    """
    cfg = config or get_config()
    if cfg.mock_llm:
        logger.info("mock_llm is True. Returning simulated JSON metadata without network call.")
        return '{"doc_type": "judgment", "doc_id": "mock_id", "claim_no": "CA 005/2025", "court": "DIFC Courts", "date": "March 15, 2025"}'

    client, model = get_llm_client_and_model(cfg)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "timeout": 15.0,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""
    except OpenAIError as exc:
        logger.error("LLM call failed for model %s: %s", model, exc)
        raise RuntimeError(f"LLM execution error ({model}): {exc}") from exc
