"""
LLM client initialisation and shared retry logic for the audit workflow.

Environment variables
---------------------
LLM_API_KEY        : required — API key for the LLM provider
LLM_BASE_URL       : optional — OpenAI-compatible base URL (default: https://api.openai.com/v1)
LLM_MODEL          : optional — model ID (default: gpt-4o-mini)
LLM_MAX_RETRIES    : optional — max retry attempts per node (default: 3)
LLM_RETRY_BASE_S   : optional — base back-off seconds (default: 2.0)
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import Any, Callable, TypeVar

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / env-var resolution
# ---------------------------------------------------------------------------

_LLM_BASE_URL: str = os.environ.get(
    "LLM_BASE_URL", "https://api.openai.com/v1"
)
_LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
_LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "3"))
_LLM_RETRY_BASE_S: float = float(os.environ.get("LLM_RETRY_BASE_S", "2.0"))


def _require_api_key() -> str:
    """Return the API key or raise early with a clear message."""
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "LLM_API_KEY is not set. "
            "Export it before running the audit worker."
        )
    return api_key


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    """
    Return a ChatOpenAI-compatible instance. Targets any OpenAI-compatible
    endpoint configured via LLM_BASE_URL.  This is intentionally not a
    singleton so callers can request different temperatures.
    """
    return ChatOpenAI(
        model=_LLM_MODEL,
        api_key=_require_api_key(),
        base_url=_LLM_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Retry decorator (async-aware, exponential back-off with full jitter)
# ---------------------------------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def with_retry(
    max_attempts: int = _LLM_MAX_RETRIES,
    base_seconds: float = _LLM_RETRY_BASE_S,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorator that retries an *async* coroutine with full-jitter exponential
    back-off.

    Back-off formula:  sleep = random.uniform(0, base * 2^attempt)
    """

    def decorator(fn: F) -> F:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts - 1:
                        cap = base_seconds * (2 ** attempt)
                        sleep = random.uniform(0, cap)
                        logger.warning(
                            "Node %s attempt %d/%d failed (%s). "
                            "Retrying in %.2fs.",
                            fn.__name__,
                            attempt + 1,
                            max_attempts,
                            type(exc).__name__,
                            sleep,
                        )
                        await asyncio.sleep(sleep)
                    else:
                        logger.error(
                            "Node %s exhausted %d retries. Last error: %s",
                            fn.__name__,
                            max_attempts,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Shared LLM prompt version metadata (injected into RefactorLLMMeta)
# ---------------------------------------------------------------------------

LLM_PROVENANCE = {
    "provider": _LLM_BASE_URL,
    "model": _LLM_MODEL,
    "prompt_version": "prompt-refactor-v1",
}

# Use function_calling for structured output — broadly supported across providers.
# All `.with_structured_output()` calls must use method="function_calling".
STRUCTURED_OUTPUT_METHOD = "function_calling"
