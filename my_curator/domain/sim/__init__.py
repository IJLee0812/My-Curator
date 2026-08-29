"""Real-to-sim mapping layer.

Scenario DNA v0.2 -> an engine-agnostic ``SimSpec`` describing how a segment should be
re-staged in CARLA 0.9.15. Stdlib only — no CARLA, no GPU, no DB imports.
"""

from my_curator.domain.sim.compilation import (
    CompilationReport,
    CompiledSegment,
    build_compilation_report,
)
from my_curator.domain.sim.coverage import CoverageReport, build_coverage_report
from my_curator.domain.sim.mapper import MappingResult, map_dna
from my_curator.domain.sim.reasons import DegradationCode, ExclusionReason, RenderFailure
from my_curator.domain.sim.render import (
    RenderOutcome,
    RenderReport,
    SegmentRef,
    build_render_report,
    select_segment,
)
from my_curator.domain.sim.road_index import RoadCandidate, RoadSelection, select_road
from my_curator.domain.sim.spec import (
    ActorSpec,
    CameraSpec,
    ControlMode,
    EgoSpec,
    SimSpec,
    WorldSpec,
)
from my_curator.domain.sim.xosc_compiler import compile_scenario

__all__ = [
    "ActorSpec",
    "CameraSpec",
    "CompilationReport",
    "CompiledSegment",
    "ControlMode",
    "CoverageReport",
    "DegradationCode",
    "EgoSpec",
    "ExclusionReason",
    "MappingResult",
    "RenderFailure",
    "RenderOutcome",
    "RenderReport",
    "RoadCandidate",
    "SegmentRef",
    "RoadSelection",
    "SimSpec",
    "WorldSpec",
    "build_compilation_report",
    "build_coverage_report",
    "build_render_report",
    "compile_scenario",
    "map_dna",
    "select_road",
    "select_segment",
]
