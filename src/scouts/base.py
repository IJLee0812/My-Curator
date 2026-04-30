from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class ScoutConfig:
    """Runtime configuration for a Scout sampling run.

    Loaded from configs/scout.yaml via from_yaml().
    n=1 activates single-call fallback (T=temperatures[0] only).
    """

    temperatures: list[float]
    seeds: dict[float, int]
    n: int
    max_tokens: int
    top_p: float
    top_k: int
    repetition_penalty: float
    engine_backend: str
    engine_base_url: str = ""
    engine_model: str = ""
    engine_timeout_s: float = 30.0
    kafka_topic_scouted: str = "curation.clip.scouted"
    kafka_topic_needs_review: str = "curation.clip.needs_review"

    @classmethod
    def from_yaml(cls, path: str) -> ScoutConfig:
        import yaml

        with open(path) as fh:
            data = yaml.safe_load(fh)
        engine = data.get("engine", {})
        kafka = data.get("kafka", {})
        seeds = {float(k): int(v) for k, v in data.get("seeds", {}).items()}
        return cls(
            temperatures=data["temperatures"],
            seeds=seeds,
            n=data["n"],
            max_tokens=data["max_tokens"],
            top_p=data["top_p"],
            top_k=data["top_k"],
            repetition_penalty=data["repetition_penalty"],
            engine_backend=engine.get("backend", "gstnvvllmvlm_api"),
            engine_base_url=engine.get("base_url", ""),
            engine_model=engine.get("model", ""),
            engine_timeout_s=float(engine.get("timeout_s", 30.0)),
            kafka_topic_scouted=kafka.get("topic_scouted", "curation.clip.scouted"),
            kafka_topic_needs_review=kafka.get("topic_needs_review", "curation.clip.needs_review"),
        )

    def seed_for(self, temperature: float) -> int:
        return self.seeds.get(temperature, 42)


@dataclass
class ScoutReport:
    """Single VLM output for one temperature sample."""

    text: str
    temperature: float
    seed: int
    latency_ms: float
    partial_sampling: bool = False


@runtime_checkable
class Scout(Protocol):
    """Structural protocol for all Scout adapters.

    Implementors must provide sample(); no base class required.
    """

    def sample(
        self,
        inputs: dict,
        prompt_config: dict,
        config: ScoutConfig,
        t0_result: str | None = None,
    ) -> list[ScoutReport]: ...
