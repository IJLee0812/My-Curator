"""Cosmos-Embed1-336p model wrapper for video-only embedding (P3-1).

Host-importable: torch and transformers are lazy-imported inside ``__init__``
so this module can be collected by pytest on a bare venv without those packages.
GPU is used at inference time when available; falls back to CPU for testing.
"""

from __future__ import annotations

from pathlib import Path

_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "hub" / "cosmos-embed1-336p"


class CosmosEmbed1:
    """Thin wrapper around Cosmos-Embed1-336p for video-only embedding.

    Args:
        model_dir: Path to the local model snapshot.  Defaults to
            ``models/hub/cosmos-embed1-336p/`` relative to the repo root.

    Usage::

        model = CosmosEmbed1()
        vec = model.embed(tensor)   # tensor: [1, 8, 3, H, W] uint8
        assert len(vec) == 768
    """

    def __init__(self, model_dir: str | Path = _MODEL_DIR) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        model_dir = Path(model_dir)
        self._processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
        model = AutoModel.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)

    def embed(self, tensor) -> list[float]:
        """Embed a single video clip.

        Args:
            tensor: ``[1, 8, 3, H, W]`` uint8 tensor on any device.

        Returns:
            768-dim L2-normalised embedding as ``list[float]``.
        """
        import torch

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        inputs = self._processor(videos=tensor)
        videos = inputs["videos"].to(self._device)
        with torch.no_grad():
            output = self._model.get_video_embeddings(videos)
        return output.visual_proj[0].float().cpu().tolist()
