"""Async vLLM OpenAI-compatible HTTP client for the Qwen3 text-only Judge critic (P4-6).

httpx-only (host-importable). One ``critique()`` is one chat completion; the caller fires N
concurrently for the self-consistency vote. Sampling defaults are the Qwen3 card values for a
quantized thinking model; the ``<think>…</think>`` block is stripped in ``domain/judge/verdict``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import httpx

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_MODEL = "qwen3-8b-awq"


class JudgeCriticError(RuntimeError):
    """Raised when a critique call fails after all retries (timeout, HTTP, or bad body)."""


@dataclass(frozen=True)
class SamplingParams:
    """Qwen3 card values for a quantized thinking model (greedy decoding avoided)."""

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    max_tokens: int = 2048


def base_url_from_env() -> str:
    """Judge critic base URL from ``JUDGE_CRITIC_URL`` (default ``http://localhost:8002``)."""
    return os.environ.get("JUDGE_CRITIC_URL", "http://localhost:8002")


class QwenTextCritic:
    """Async client for the judge-critic vLLM server (OpenAI ``/v1/chat/completions``)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        sampling: SamplingParams | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        retries: int = 1,
        client: httpx.AsyncClient | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        base = (base_url or base_url_from_env()).rstrip("/")
        self._url = f"{base}/v1/chat/completions"
        self._model = model
        self._sampling = sampling or SamplingParams()
        self._retries = max(0, int(retries))
        # An injected client is caller-owned; one we build (optionally over an injected
        # transport, for tests) is owned and closed by aclose() so no event loop leaks.
        if client is not None:
            self._client = client
            self._owns_client = False
        else:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), transport=transport)
            self._owns_client = True

    async def critique(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw assistant message for one (system, user) pair; retry then raise."""
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **asdict(self._sampling),
        }
        last_exc: Exception | None = None
        for _ in range(self._retries + 1):
            try:
                resp = await self._client.post(self._url, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                last_exc = exc
        raise JudgeCriticError(
            f"critique failed after {self._retries + 1} attempt(s): {last_exc}"
        ) from last_exc

    async def aclose(self) -> None:
        """Close the underlying client if this instance created it."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> QwenTextCritic:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
