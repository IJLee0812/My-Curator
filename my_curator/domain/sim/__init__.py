"""Real-to-sim mapping layer.

Scenario DNA v0.2 -> an engine-agnostic ``SimSpec`` describing how a segment should be
re-staged in CARLA 0.9.15. Stdlib only — no CARLA, no GPU, no DB imports.
"""

from my_curator.domain.sim.reasons import DegradationCode, ExclusionReason
from my_curator.domain.sim.spec import (
    ActorSpec,
    CameraSpec,
    ControlMode,
    EgoSpec,
    SimSpec,
    WorldSpec,
)

__all__ = [
    "ActorSpec",
    "CameraSpec",
    "ControlMode",
    "DegradationCode",
    "EgoSpec",
    "ExclusionReason",
    "SimSpec",
    "WorldSpec",
]
