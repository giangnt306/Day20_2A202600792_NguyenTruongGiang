"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
All retry, timeout, token logging, and cost accounting live here, not inside agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# USD per 1K tokens. Extend this table as you add models.
_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.00250, 0.01000),
    "gpt-4.1-mini": (0.00040, 0.00160),
}


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate USD cost from a known price table; None if model is unknown."""

    price = _PRICE_PER_1K.get(model)
    if price is None:
        return None
    in_rate, out_rate = price
    return (input_tokens / 1000) * in_rate + (output_tokens / 1000) * out_rate


class LLMClient:
    """Provider-agnostic LLM client.

    Uses OpenAI when an API key is configured, otherwise falls back to a deterministic
    mock so tests and offline CI keep working without network access.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.openai_model
        self._client = None
        if self.settings.llm_enabled:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                timeout=self.settings.timeout_seconds,
            )

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Return a model completion with retry, cost, and token logging."""

        if self._client is None:
            return self._mock_complete(system_prompt, user_prompt)
        return self._complete_with_retry(
            system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def _complete_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        assert self._client is not None
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = getattr(usage, "prompt_tokens", None)
        output_tokens = getattr(usage, "completion_tokens", None)
        cost = (
            estimate_cost(self.model, input_tokens, output_tokens)
            if input_tokens is not None and output_tokens is not None
            else None
        )
        logger.info(
            "llm.complete model=%s in_tokens=%s out_tokens=%s cost_usd=%s",
            self.model,
            input_tokens,
            output_tokens,
            f"{cost:.6f}" if cost is not None else "n/a",
        )
        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def _mock_complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Deterministic offline stand-in used when no API key is configured."""

        logger.warning("llm.complete using MOCK provider (no OPENAI_API_KEY configured)")
        content = (
            "[MOCK LLM OUTPUT]\n"
            f"system: {system_prompt[:80].strip()}\n"
            f"task: {user_prompt[:200].strip()}\n"
            "Provide OPENAI_API_KEY in .env to enable real completions."
        )
        approx_in = len(system_prompt + user_prompt) // 4
        approx_out = len(content) // 4
        return LLMResponse(
            content=content,
            input_tokens=approx_in,
            output_tokens=approx_out,
            cost_usd=0.0,
        )
