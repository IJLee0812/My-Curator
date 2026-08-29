"""Stage one compiled scenario and produce its videos. Runs inside the simulator image.

This is the py3.7 half of a render: the CARLA API, the cameras and every GStreamer
pipeline live here, because none of them exist in the host environment. It takes a
scenario the host compiled, executes it, and writes three MP4s plus a JSON summary on
stdout for the host to read back.

    python3.7 -m my_curator.cli.run_sim_stage \
        --scenario /opt/sim/<clip>.xosc --cameras /opt/sim/<clip>.cameras.json \
        --out-dir /opt/render/<clip> --source /video/<clip>.mp4 \
        --source-start-s 4.13 --duration-s 5.0

Never invoked directly in normal use — ``run_sim_render`` drives it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from my_curator.adapters.sim import encoder
from my_curator.adapters.sim.carla_executor import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    ExecutionError,
    ScenarioExecutor,
    apply_weather,
    connect,
    ensure_town,
)
from my_curator.adapters.sim.recorder import DualViewRecorder, RecordingError
from my_curator.adapters.sim.xosc_reader import read_program
from my_curator.domain.sim.program import UnsupportedScenarioError
from my_curator.domain.sim.reasons import RenderFailure

log = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Execute one compiled scenario and record it")
    p.add_argument("--scenario", required=True, help="compiled .xosc to execute")
    p.add_argument("--cameras", required=True, help="camera rig JSON written beside the scenario")
    p.add_argument("--out-dir", required=True, help="writable directory for frames and videos")
    p.add_argument("--duration-s", type=float, required=True, help="recorded segment length")
    p.add_argument("--source", help="original clip, for the comparison view")
    p.add_argument("--source-start-s", type=float, default=0.0, help="segment start in the source")
    p.add_argument("--risk-level", default="unknown", help="burned into the overlay")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--keep-frames", action="store_true", help="do not delete raw frames")
    return p


def _overlay_text(clip_id: str, risk_level: str) -> str:
    return f"{risk_level} | {clip_id[:8]}"


class _Kinematics:
    """Per-tick measurements over the recorded window, summarised for the render report.

    The videos are the deliverable but cannot be asserted on; these numbers can.
    ``frontal_visible_ratio`` is the share of recorded ticks with an adversary inside a
    ~60° cone ahead of ego within 50 m. It is low by construction for encounters that
    *complete* — a head-on pass, a pedestrian finishing a crossing — so the gate is
    ``approach_visible_ratio``, the same measure up to the closest approach only: an actor
    invisible on the way in was staged wrong.
    """

    def __init__(self, executor: ScenarioExecutor, ego_name: str) -> None:
        self._executor = executor
        self._ego_name = ego_name
        self._samples = 0
        self._visible = 0
        self._visible_flags: list = []
        self._gaps: list = []
        self._ego_speeds: list = []

    def sample(self) -> None:
        entities = self._executor.staged_entities()
        ego = entities[self._ego_name]
        location = ego.actor.get_location()
        forward = ego.actor.get_transform().get_forward_vector()
        velocity = ego.actor.get_velocity()
        self._ego_speeds.append((velocity.x**2 + velocity.y**2) ** 0.5)
        nearest = None
        visible = False
        for entity in entities.values():
            if entity.is_ego:
                continue
            other = entity.actor.get_location()
            dx, dy = other.x - location.x, other.y - location.y
            distance = (dx * dx + dy * dy) ** 0.5
            if nearest is None or distance < nearest:
                nearest = distance
            if 0.5 < distance < 50.0 and (dx * forward.x + dy * forward.y) / distance > 0.5:
                visible = True
        self._samples += 1
        self._visible_flags.append(visible)
        if visible:
            self._visible += 1
        if nearest is not None:
            self._gaps.append(nearest)

    def summary(self) -> dict:
        if not self._samples:
            return {}
        out = {
            "ego_v_start_mps": round(self._ego_speeds[0], 2),
            "ego_v_end_mps": round(self._ego_speeds[-1], 2),
            "ego_v_mean_mps": round(sum(self._ego_speeds) / len(self._ego_speeds), 2),
        }
        if self._gaps:
            closest = self._gaps.index(min(self._gaps))
            approach = self._visible_flags[: closest + 1]
            out.update(
                {
                    "gap_start_m": round(self._gaps[0], 1),
                    "gap_min_m": round(min(self._gaps), 1),
                    "gap_end_m": round(self._gaps[-1], 1),
                    "frontal_visible_ratio": round(self._visible / self._samples, 2),
                    "approach_visible_ratio": round(sum(approach) / len(approach), 2),
                }
            )
        return out


def _record(program, cameras, args) -> tuple[list, list[str], dict]:
    fps = int(cameras[0]["fps"])
    tick_s = 1.0 / fps
    # Counted in ticks, not compared against a clock: a clock comparison captures one
    # frame too many whenever the segment is not a whole number of ticks long.
    warmup_ticks = int(round(max(0.0, program.stop_time_s - args.duration_s) / tick_s))
    expected_frames = int(round(args.duration_s * fps))
    total_ticks = warmup_ticks + expected_frames

    _client, world = connect(args.host, args.port)
    ensure_town(world, program.town)
    apply_weather(world, program.environment.to_weather())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fired: list[str] = []
    with ScenarioExecutor(world, program, tick_s=tick_s) as executor:
        executor.stage()
        ego = executor.entity(program.ego.name).actor
        kinematics = _Kinematics(executor, program.ego.name)
        with DualViewRecorder(world, ego, cameras, out_dir) as recorder:
            for index in range(total_ticks):
                elapsed_s = index * tick_s
                recording = index >= warmup_ticks
                fired.extend(executor.step(elapsed_s, fire=recording))
                if recording:
                    kinematics.sample()
                recorder.tick(record=recording)
            views = recorder.views(expected_frames)
    return views, fired, kinematics.summary()


def _encode(program, views, cameras, args) -> dict[str, str]:
    out_dir = Path(args.out_dir)
    fps = int(cameras[0]["fps"])
    outputs: dict[str, str] = {}
    overlay = _overlay_text(program.clip_id, args.risk_level)

    for view in views:
        target = out_dir / f"{view.view}.mp4"
        encoder.run_pipeline(
            encoder.view_pipeline(
                view.path,
                target,
                width=view.width,
                height=view.height,
                fps=fps,
                overlay=overlay,
            ),
            what=f"encode {view.view}",
        )
        outputs[view.view] = str(target)

    if args.source:
        outputs["compare"] = str(_compare(views, args, fps))
    return outputs


def _compare(views, args, fps: int) -> Path:
    out_dir = Path(args.out_dir)
    source = Path(args.source)
    if not source.is_file():
        raise encoder.EncodingError(f"source clip {source} is not readable")

    decoded = out_dir / "original_full.bgra"
    encoder.run_pipeline(encoder.extract_pipeline(source, decoded, fps), what="decode source clip")
    segment = out_dir / "original.bgra"
    encoder.slice_raw(
        decoded, segment, start_s=args.source_start_s, duration_s=args.duration_s, fps=fps
    )
    decoded.unlink()

    by_view = {view.view: view for view in views}
    target = out_dir / "compare.mp4"
    encoder.run_pipeline(
        encoder.compare_pipeline(
            segment,
            by_view["ego"].path,
            by_view["chase"].path,
            target,
            width=by_view["ego"].width,
            height=by_view["ego"].height,
            fps=fps,
        ),
        what="compose comparison",
    )
    return target


def _discard_frames(out_dir: Path) -> None:
    for raw in out_dir.glob("*.bgra"):
        raw.unlink()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir)

    try:
        program = read_program(args.scenario)
        cameras = json.loads(Path(args.cameras).read_text())
        views, fired, kinematics = _record(program, cameras, args)
        outputs = _encode(program, views, cameras, args)
    except UnsupportedScenarioError as exc:
        return _fail(RenderFailure.SCENARIO_UNSUPPORTED, str(exc))
    except ExecutionError as exc:
        return _fail(exc.reason, str(exc))
    except RecordingError as exc:
        return _fail(RenderFailure.SENSOR_FRAMES_LOST, str(exc))
    except encoder.EncodingError as exc:
        reason = (
            RenderFailure.SOURCE_VIDEO_UNREADABLE
            if "source clip" in str(exc)
            else RenderFailure.ENCODING_FAILED
        )
        return _fail(reason, str(exc))
    except Exception as exc:  # noqa: BLE001 — an unrecorded failure is worse than a broad catch
        log.exception("staging failed")
        return _fail(RenderFailure.STAGE_CRASHED, f"{type(exc).__name__}: {exc}")
    finally:
        if not args.keep_frames and out_dir.is_dir():
            _discard_frames(out_dir)

    print(
        json.dumps(
            {
                "status": "rendered",
                "clip_id": program.clip_id,
                "town": program.town,
                "frames": {view.view: view.frames for view in views},
                "events_fired": fired,
                "kinematics": kinematics,
                "outputs": outputs,
            }
        )
    )
    return 0


def _fail(reason: RenderFailure, message: str) -> int:
    log.error("%s: %s", reason.value, message)
    print(json.dumps({"status": "failed", "failure_reason": reason.value, "detail": message}))
    return 1


if __name__ == "__main__":
    sys.exit(main())
