"""Cosmos-Embed1-336p text + video encoder for curation-api (P3-2).

Host-importable: torch and transformers are lazy-imported inside __init__
so this module is collectable by pytest on a bare venv.
GPU is the cuda:0 device visible inside the container
(physical GPU 1 via CUDA_VISIBLE_DEVICES=0 in compose).
"""

from __future__ import annotations

from pathlib import Path

_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "hub" / "cosmos-embed1-336p"


class CosmosEmbed1Encoder:
    """Cosmos-Embed1-336p text tower + video tower wrapper.

    Both towers output 768-dim L2-normalised vectors; raw IP score equals
    cosine similarity.  logit_scale is NOT applied here — it is a
    zero-shot classification parameter only (confirmed: official Cosmos-Embed1
    model card and NVIDIA NIM docs).
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

    def encode_text(self, text: str) -> list[float]:
        """Embed a natural-language query via the text tower.

        Returns:
            768-dim L2-normalised embedding as list[float].
        """
        import torch

        inputs = self._processor(text=[text], return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            output = self._model.get_text_embeddings(**inputs)
        return output.text_proj[0].float().cpu().tolist()

    def encode_video(self, tensor) -> list[float]:
        """Embed a video clip via the video tower.

        Args:
            tensor: [1, 8, 3, H, W] uint8 tensor on any device.

        Returns:
            768-dim L2-normalised embedding as list[float].
        """
        import torch

        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        inputs = self._processor(videos=tensor)
        videos = inputs["videos"].to(self._device)
        # Cosmos-Embed1 ViT is mixed precision (Conv2d/Linear bfloat16,
        # LayerNorm float32).  autocast inserts dtype conversions at layer
        # boundaries; manual casts cause downstream layer-norm mismatches.
        with torch.no_grad():
            if self._device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = self._model.get_video_embeddings(videos)
            else:
                output = self._model.get_video_embeddings(videos)
        return output.visual_proj[0].float().cpu().tolist()
