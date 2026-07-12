"""Persistent vLLM engine holder decoupled from the GStreamer element lifecycle.

Owns the ``vllm.LLM`` + tokenizer so they can outlive per-clip pipelines and be
reused across a session (see ``run_pipeline --warm``). Gst-free; torch/vllm are
imported lazily inside ``load()`` so the module stays importable on hosts without
CUDA.
"""

from __future__ import annotations

import threading


class VLMEngine:
    """Loads and owns the vLLM model + tokenizer for reuse across pipelines."""

    def __init__(
        self,
        model,
        max_model_len,
        trust_remote_code,
        gpu_memory_utilization,
        enforce_eager,
        gpu_id=-1,
    ):
        self.model = model
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enforce_eager = enforce_eager
        self.gpu_id = gpu_id
        self.llm = None
        self.tokenizer = None
        # Serializes generate() so a warm-mode worker orphaned by one clip can't
        # race the next clip's worker on the non-thread-safe vllm.LLM (kills EngineCore).
        self._generate_lock = threading.Lock()

    @classmethod
    def from_config(cls, config):
        return cls(
            model=config.model_path,
            max_model_len=config.max_model_len,
            trust_remote_code=config.trust_remote_code,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            gpu_id=config.gpu_id,
        )

    @property
    def is_loaded(self) -> bool:
        return self.llm is not None

    def load(self) -> None:
        if self.llm is not None:
            return

        import torch
        from transformers import AutoTokenizer
        from vllm import LLM

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        torch.cuda.init()
        dev = self.gpu_id if 0 <= self.gpu_id < torch.cuda.device_count() else 0
        torch.cuda.set_device(dev)
        print(f"VLMEngine: CUDA on GPU {dev} ({torch.cuda.get_device_name(dev)})")

        llm_kwargs = dict(
            model=self.model,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        if self.enforce_eager:
            llm_kwargs["enforce_eager"] = True
        self.llm = LLM(**llm_kwargs)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model, trust_remote_code=self.trust_remote_code
            )
        except Exception:
            self.tokenizer = None
        print(f"VLMEngine: model loaded ({self.model})")

    def generate(self, inputs, sampling_params):
        with self._generate_lock:
            return self.llm.generate(inputs, sampling_params=sampling_params)

    def shutdown(self) -> None:
        if self.llm is None:
            return
        try:
            if hasattr(self.llm, "shutdown"):
                self.llm.shutdown()
            elif hasattr(self.llm, "llm_engine") and hasattr(self.llm.llm_engine, "shutdown"):
                self.llm.llm_engine.shutdown()
        except Exception as e:
            print(f"VLMEngine: shutdown error: {e}")
        finally:
            self.llm = None
            self.tokenizer = None
