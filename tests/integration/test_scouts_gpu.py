"""GPU live-engine smoke test for CosmosReasonScout via NvVllmVLM.get_llm().

Requires:
  - @pytest.mark.gpu — GPU CI runner or manual run with --run-gpu flag.
  - DS 9.0 container with CUDA available and Cosmos-Reason2-8B FP8 model loaded.
  - configs/config_driving_scene.yaml present (for model path / params).

Run inside container:
    pytest tests/integration/test_scouts_gpu.py -m gpu -v
"""

import os
import sys

import pytest


@pytest.mark.gpu
@pytest.mark.integration
def test_cosmos_reason_scout_gpu_smoke():
    """Live smoke: CosmosReasonScout returns ≥1 ScoutReport via NvVllmVLM.get_llm().

    Flow:
      1. Register NvVllmVLM as GStreamer element (triggers vLLM model load).
      2. Call element.get_llm() — verifies P2-4 accessor wiring.
      3. Run CosmosReasonScout.sample() with a synthetic black frame.
      4. Assert ≥1 report with non-negative latency.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    gi = pytest.importorskip("gi")
    gi.require_version("Gst", "1.0")  # type: ignore[attr-defined]
    import numpy as np
    from gi.repository import Gst  # type: ignore[import]

    Gst.init(None)
    from my_curator.adapters.gst.nvvllmvlm import NvVllmVLM  # noqa: PLC0415

    Gst.Element.register(None, "nvvllmvlm", Gst.Rank.NONE, NvVllmVLM)

    # Creating the element triggers __init__ which loads the vLLM model
    element = Gst.ElementFactory.make("nvvllmvlm", "smoke-vlm")
    assert element is not None, "Failed to create NvVllmVLM GStreamer element"

    # P2-4 acceptance criterion: get_llm() returns the live LLM instance
    llm = element.get_llm()
    assert llm is not None, (
        "get_llm() returned None — vLLM model not loaded. "
        "Ensure the Cosmos-Reason2-8B FP8 checkpoint is present."
    )

    from PIL import Image as PILImage

    from my_curator.adapters.scout.cosmos_reason import CosmosReasonScout
    from my_curator.domain.scout.base import ScoutConfig

    scout = CosmosReasonScout(llm=llm)
    config = ScoutConfig.from_yaml("configs/scout.yaml")

    # Minimal synthetic frame: 224×224 black image, converted to PIL
    frame_np = np.zeros((224, 224, 3), dtype=np.uint8)
    frame_pil = PILImage.fromarray(frame_np)

    # Build chat-template formatted prompt with <image> token so vLLM can
    # apply the multimodal placeholder replacement (required by Qwen3VL).
    tokenizer = element.tokenizer
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this driving scene frame."},
            ],
        }
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = {
        "prompt": prompt_text,
        "multi_modal_data": {"image": [frame_pil]},
    }

    reports = scout.sample(inputs, {}, config, t0_result=None)

    assert len(reports) >= 1, f"Expected ≥1 ScoutReport, got {len(reports)}"
    assert all(isinstance(r.text, str) for r in reports), "Report text must be str"
    assert reports[0].latency_ms >= 0, "latency_ms must be non-negative"
