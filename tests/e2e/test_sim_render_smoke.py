"""Render smoke: one curated segment becomes three videos of the right length.

The compilation smoke (``test_sim_scenario_smoke``) proves a scenario is readable. This one
proves it *runs*: the simulator is booted on the town the scenario names, entities are
staged, two cameras record, and the encoder produces an ego view, a chase view and the
side-by-side comparison against the source clip.

The render tool owns the simulator's lifecycle here — it boots the town it needs, because
switching maps at runtime segfaults this build. Expect the container to be restarted.

Requires Postgres, a built road index, and the simulate profile's image. CARLA and
judge-critic cannot share GPU 0 — bring the judge down first.

Run: ``pytest tests/e2e/test_sim_render_smoke.py -m gpu``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from my_curator.adapters.sim.encoder import PANE_HEIGHT, PANE_WIDTH
from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.domain.scout.versioning import CURRENT_DNA_VERSION
from my_curator.domain.sim.render import RENDERED
from my_curator.domain.sim.spec import RENDER_FPS, RENDER_HEIGHT, RENDER_WIDTH

pytestmark = [pytest.mark.gpu, pytest.mark.e2e]

CONTAINER = "my-curator-carla"
IMAGE = "my-curator-carla:0.9.15"
RENDER_TIMEOUT_S = 900


def _skip_unless_ready() -> None:
    if not shutil.which("docker"):
        pytest.skip("docker unavailable")
    if "PG_USER" not in os.environ:
        pytest.skip("Postgres environment not configured (.env not loaded)")
    probe = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "which", IMAGE, "gst-launch-1.0"],
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("simulator image predates the render tooling — rebuild carla-server")


async def _pick_segment() -> tuple[str, float] | None:
    """The most severe segment in the corpus, with the length it must reproduce."""
    repo = await PGRepository.create(dsn_from_env())
    try:
        rows = await repo.list_sim_segments(dna_version=CURRENT_DNA_VERSION, limit=5000)
        index = await repo.list_sim_road_index()
    finally:
        await repo.close()
    if not rows or not index:
        return None
    ranked = sorted(
        rows,
        key=lambda r: {"critical": 0, "elevated": 1, "nominal": 2}.get(
            (r["dna_json"].get("planner_logic") or {}).get("risk_level"), 3
        ),
    )
    chosen = ranked[0]
    return chosen["source_clip_id"], float(chosen["end_s"]) - float(chosen["start_s"])


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    _skip_unless_ready()
    try:
        picked = asyncio.run(_pick_segment())
    except Exception as exc:  # noqa: BLE001 — an unreachable database is a skip
        pytest.skip(f"Postgres unreachable: {exc}")
    if picked is None:
        pytest.skip("no curated segments, or sim_road_index is empty")

    source_clip_id, duration_s = picked
    render_dir = tmp_path_factory.mktemp("render")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "my_curator.cli.run_sim_render",
            "--clip",
            source_clip_id,
            "--render-dir",
            str(render_dir),
            "--no-upload",
            "--report",
            str(render_dir / "report.json"),
        ],
        capture_output=True,
        text=True,
        timeout=RENDER_TIMEOUT_S,
        check=False,
    )
    report_path = render_dir / "report.json"
    if not report_path.is_file():
        pytest.fail(f"render produced no report:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    return json.loads(report_path.read_text()), render_dir, duration_s


def _decoded(path: Path) -> tuple[int, int, int]:
    """Decode a video in the simulator image and return (width, height, frames).

    The host has neither an H.264 decoder nor an encoder, so anything that inspects a
    rendered video has to run where the video was made. Frames are counted from the raw
    byte size rather than parsed out of a report: it is exact, and it needs no extra tool.
    """
    script = (
        f"gst-launch-1.0 -v filesrc location=/probe/{path.name} ! decodebin ! videoconvert "
        "! fakesink 2>&1 | grep -oE 'width=\\(int\\)[0-9]+, height=\\(int\\)[0-9]+' "
        "| head -1; "
        f"gst-launch-1.0 -q filesrc location=/probe/{path.name} ! decodebin ! videoconvert "
        "! video/x-raw,format=BGRA ! filesink location=/tmp/p.raw 2>/dev/null; "
        "stat -c %s /tmp/p.raw"
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{path.parent}:/probe:ro",
            "--entrypoint",
            "bash",
            IMAGE,
            "-lc",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    caps, _, size = result.stdout.strip().rpartition("\n")
    numbers = re.findall(r"\d+", caps)
    if len(numbers) < 2 or not size.strip().isdigit():
        raise AssertionError(
            f"could not decode {path.name}: {result.stdout} {result.stderr[-400:]}"
        )
    width, height = int(numbers[0]), int(numbers[1])
    return width, height, int(size) // (width * height * 4)


def test_the_segment_renders(rendered):
    report, _, _ = rendered
    assert report["attempted"] == 1
    assert report["rendered"] == 1, report["failures"]


def test_all_three_videos_are_produced(rendered):
    _, render_dir, _ = rendered
    videos = sorted(p.name for p in render_dir.rglob("*.mp4"))
    assert videos == ["chase.mp4", "compare.mp4", "ego.mp4"]
    for video in render_dir.rglob("*.mp4"):
        assert video.stat().st_size > 10_000, f"{video.name} is suspiciously small"


def test_raw_frames_are_not_left_behind(rendered):
    """~370 MB per render; leaving them would fill the disk in a batch."""
    _, render_dir, _ = rendered
    assert list(render_dir.rglob("*.mp4")), "nothing rendered, so this proves nothing"
    assert list(render_dir.rglob("*.bgra")) == []


def test_the_simulator_views_have_the_declared_geometry(rendered):
    _, render_dir, _ = rendered
    for view in ("ego", "chase"):
        width, height, _ = _decoded(next(render_dir.rglob(f"{view}.mp4")))
        assert (width, height) == (RENDER_WIDTH, RENDER_HEIGHT)


def test_the_comparison_is_three_panes_wide(rendered):
    _, render_dir, _ = rendered
    width, height, _ = _decoded(next(render_dir.rglob("compare.mp4")))
    assert (width, height) == (PANE_WIDTH * 3, PANE_HEIGHT)


def test_every_view_holds_exactly_the_source_segment(rendered):
    """Frame-for-frame alignment is the point of the duration correction."""
    _, render_dir, duration_s = rendered
    expected = round(duration_s * RENDER_FPS)
    for view in ("ego", "chase", "compare"):
        _, _, frames = _decoded(next(render_dir.rglob(f"{view}.mp4")))
        assert frames == expected, view


def test_the_attempt_is_recorded_in_the_ledger(rendered):
    report, _, _ = rendered

    async def _last():
        repo = await PGRepository.create(dsn_from_env())
        try:
            return await repo.list_sim_renders(limit=1)
        finally:
            await repo.close()

    rows = asyncio.run(_last())
    assert rows and rows[0]["status"] == RENDERED
    assert rows[0]["town"] == report["towns"][0]["town"]
