from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from src.scouts.base import ScoutConfig, ScoutReport

if TYPE_CHECKING:
    pass


class CosmosReasonScout:
    """Scout adapter for Cosmos-Reason2-8B FP8.

    Spike result (2026-04-29): vLLM runs as in-process vllm.LLM (synchronous).
    No HTTP endpoint is exposed. Engine is injected at construction time so
    unit tests can supply a mock without requiring vllm or GPU.

    Integration flow (wired in P2-4):
      1. nvvllmvlm worker thread calls llm.generate(inputs, T=0.3) and emits
         vlm-result signal → caller passes that text as t0_result.
      2. sample() wraps t0_result as ScoutReport(T=0.3).
      3. sample() batch-generates T=0.5 + T=0.7 via a single llm.generate()
         call with two SamplingParams (prefill KV shared, ~130 MB overhead).
      4. Returns list[ScoutReport] of length n (or <n on partial failure).
    """

    def __init__(
        self,
        llm: Any | None = None,
        _sampling_params_cls: Any | None = None,
    ) -> None:
        self._llm = llm
        # _sampling_params_cls is injected in unit tests to avoid vllm import.
        # In production it is resolved lazily from vllm on first use.
        self._SamplingParams = _sampling_params_cls

    # ------------------------------------------------------------------
    # Public API (Scout Protocol)
    # ------------------------------------------------------------------

    def sample(
        self,
        inputs: dict,
        prompt_config: dict,
        config: ScoutConfig,
        t0_result: str | None = None,
    ) -> list[ScoutReport]:
        """Return up to config.n ScoutReports for the given inputs.

        If t0_result is supplied (pre-computed by nvvllmvlm at T=temperatures[0]),
        it is wrapped directly — no additional generate() call for T[0].
        Extra temperatures are batch-generated in one llm.generate() call.

        On partial failure (batch raises or llm is None), returns whatever
        succeeded and sets partial_sampling=True on all reports.
        """
        reports: list[ScoutReport] = []
        temperatures = config.temperatures[: config.n]

        t0 = temperatures[0]
        if t0_result is not None:
            reports.append(
                ScoutReport(
                    text=t0_result,
                    temperature=t0,
                    seed=config.seed_for(t0),
                    latency_ms=0.0,
                )
            )
        else:
            r = self._generate_one(inputs, t0, config.seed_for(t0), config)
            if r:
                reports.append(r)

        if config.n > 1 and len(temperatures) > 1:
            extra = self._generate_batch(inputs, temperatures[1:], config)
            reports.extend(extra)

        if len(reports) < config.n:
            for r in reports:
                r.partial_sampling = True

        return reports

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_sampling_params(self) -> Any | None:
        if self._SamplingParams is not None:
            return self._SamplingParams
        try:
            from vllm import SamplingParams

            self._SamplingParams = SamplingParams
            return SamplingParams
        except ImportError:
            return None

    def _generate_batch(
        self,
        inputs: dict,
        temperatures: list[float],
        config: ScoutConfig,
    ) -> list[ScoutReport]:
        """Batch-submit multiple temperatures in a single llm.generate() call."""
        if not temperatures or self._llm is None:
            return []

        SamplingParams = self._resolve_sampling_params()
        if SamplingParams is None:
            return []

        params_list = [
            SamplingParams(
                temperature=t,
                seed=config.seed_for(t),
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
            )
            for t in temperatures
        ]
        prompts = [inputs] * len(temperatures)

        reports: list[ScoutReport] = []
        try:
            t_start = time.perf_counter()
            outputs = self._llm.generate(prompts, sampling_params=params_list)
            per_ms = (time.perf_counter() - t_start) * 1000.0 / max(len(temperatures), 1)
            for out, temp in zip(outputs, temperatures, strict=False):
                text = out.outputs[0].text if out.outputs else ""
                reports.append(
                    ScoutReport(
                        text=text,
                        temperature=temp,
                        seed=config.seed_for(temp),
                        latency_ms=per_ms,
                    )
                )
        except Exception:
            pass

        return reports

    def _generate_one(
        self,
        inputs: dict,
        temperature: float,
        seed: int,
        config: ScoutConfig,
    ) -> ScoutReport | None:
        """Single-temperature call — used when t0_result is not pre-computed."""
        if self._llm is None:
            return None

        SamplingParams = self._resolve_sampling_params()
        if SamplingParams is None:
            return None

        params = SamplingParams(
            temperature=temperature,
            seed=seed,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
        )
        try:
            t_start = time.perf_counter()
            outputs = self._llm.generate([inputs], sampling_params=params)
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
            return ScoutReport(text=text, temperature=temperature, seed=seed, latency_ms=latency_ms)
        except Exception:
            return None
