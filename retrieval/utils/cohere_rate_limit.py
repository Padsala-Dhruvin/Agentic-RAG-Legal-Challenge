"""
retrieval/utils/cohere_rate_limit.py

Rate-Limited Cohere Embeddings Wrapper (Sub-Step 5.2).
Provides resilient batching and exponential backoff retry logic
when embedding legal chunks via Cohere's cloud API (`embed-english-v3.0`).
"""

import logging
import time
from typing import Any, List

logger = logging.getLogger(__name__)


def embed_with_backoff(
    client: Any,
    texts: List[str],
    model: str = "embed-english-v3.0",
    input_type: str = "search_document",
    max_retries: int = 5,
    initial_backoff: float = 2.0,
) -> List[List[float]]:
    """Call Cohere client.embed with automatic retry on rate limits (429/timeout)."""
    if not texts:
        return []

    backoff = initial_backoff
    for attempt in range(1, max_retries + 1):
        try:
            response = client.embed(
                texts=texts,
                model=model,
                input_type=input_type,
            )
            # Return list of float vectors
            return [list(map(float, vec)) for vec in response.embeddings]
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "timeout" in err_str:
                if attempt == max_retries:
                    logger.error("Max retries (%d) reached for Cohere embed. Raising exception.", max_retries)
                    raise
                logger.warning(
                    "Cohere rate limit hit (attempt %d/%d). Sleeping for %.1f seconds... (%s)",
                    attempt,
                    max_retries,
                    backoff,
                    e,
                )
                time.sleep(backoff)
                backoff *= 2.0  # Exponential backoff
            else:
                logger.error("Non-retriable Cohere embed error: %s", e)
                raise

    return []


def batch_embed_with_backoff(
    client: Any,
    texts: List[str],
    model: str = "embed-english-v3.0",
    input_type: str = "search_document",
    batch_size: int = 96,
    sleep_between_batches: float = 0.2,
    max_retries: int = 5,
) -> List[List[float]]:
    """Embed large lists of text chunks by slicing into Cohere's max batch size (96)."""
    if not texts:
        return []

    all_embeddings: List[List[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info("Batch embedding %d texts across %d batches (batch_size=%d)...", len(texts), total_batches, batch_size)

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.debug("Processing Cohere embed batch %d/%d (%d items)...", batch_num, total_batches, len(batch))

        batch_vecs = embed_with_backoff(
            client=client,
            texts=batch,
            model=model,
            input_type=input_type,
            max_retries=max_retries,
        )
        all_embeddings.extend(batch_vecs)

        if i + batch_size < len(texts) and sleep_between_batches > 0:
            time.sleep(sleep_between_batches)

    return all_embeddings
