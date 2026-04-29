"""GPU live-engine smoke test for CosmosReasonScout.

Requires:
  - @pytest.mark.gpu — GPU CI runner or manual run with --run-gpu flag.
  - Running DS pipeline with NvVllmVLM element (P2-4 integration).

Until P2-4 wires the llm reference, this test is skipped with a clear message.
"""

import pytest

from src.scouts.base import ScoutConfig


@pytest.mark.gpu
@pytest.mark.integration
def test_cosmos_reason_scout_gpu_smoke():
    """Live smoke: CosmosReasonScout returns ≥1 ScoutReport on real vLLM engine.

    Skipped until P2-4 wires NvVllmVLM.get_llm() into the Scout adapter.
    """
    pytest.skip(
        "GPU smoke test requires running DS pipeline with NvVllmVLM.get_llm() "
        "(wired in P2-4). Re-enable after P2-4 merge."
    )

    # --- Template (activate post P2-4) ---
    # from src.scouts.cosmos_reason import CosmosReasonScout
    # llm = get_llm_from_pipeline()  # P2-4 helper
    # scout = CosmosReasonScout(llm=llm)
    # config = ScoutConfig.from_yaml("configs/scout.yaml")
    # inputs = build_test_inputs()   # P2-4 helper
    #
    # reports = scout.sample(inputs, {}, config, t0_result=None)
    #
    # assert len(reports) >= 1
    # assert all(r.text for r in reports)
    # assert reports[0].latency_ms > 0
