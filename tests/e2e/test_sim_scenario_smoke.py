"""Interoperability smoke: CARLA's own scenario_runner accepts what the compiler emits.

Passing the vendored XSD proves the document is well-formed OpenSCENARIO 1.0. It does not
prove a real reader accepts it — readers implement subsets, and ours has to survive one
in P5-4. This closes that gap early, in two stages:

* **parse** — scenario_runner's pinned validator (xmlschema 1.0.18) checks the document.
  Needs no simulator, so a disagreement with the host's newer validator surfaces here.
* **load** — ``OpenScenarioConfiguration`` against a live server: the road network is
  resolved and the entities are built, which is the first point anything CARLA-specific
  can fail.

The scenario is compiled for whichever town the server is already running, never for a
different one. Switching maps at runtime segfaults this CARLA build — a bare
``client.load_world()`` takes the server down with signal 11 — so a test that forced a
switch would kill the shared simulator rather than report a defect. Boot the server on the
town you want with ``CARLA_MAP``.

Requires the simulate profile:
    docker compose --env-file .env -f infra/compose.base.yml -f infra/compose.simulate.yml \\
        --profile simulate up -d carla-server

Auto-skips when the container is absent or the scenario mount is not visible inside it.
CARLA and judge-critic cannot share GPU 0 — bring the judge down first.

Run: ``pytest tests/e2e/test_sim_scenario_smoke.py -m gpu``.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from my_curator.adapters.sim.xodr_parser import parse_town
from my_curator.adapters.sim.xosc_writer import serialize, validate
from my_curator.domain.sim.road_index import select_road
from my_curator.domain.sim.spec import (
    ActorSpec,
    CameraSpec,
    ControlMode,
    EgoSpec,
    RoadQuery,
    SafetyEventSpec,
    SimSpec,
    WorldSpec,
)
from my_curator.domain.sim.xosc_compiler import compile_scenario

pytestmark = [pytest.mark.gpu, pytest.mark.e2e]

CONTAINER = "my-curator-carla"
MOUNT = "/opt/sim"

SCENARIO_NAME = "smoke_reconstruction.xosc"
OPENDRIVE_DIR = "/home/carla/CarlaUE4/Content/Carla/Maps/OpenDrive"


def _host_dir() -> Path | None:
    """Host side of the /opt/sim mount, resolved the same way compose resolves it."""
    explicit = os.environ.get("SIM_ARTIFACT_DIR")
    if explicit:
        return Path(explicit)
    data_root = os.environ.get("DATA_ROOT")
    return Path(data_root) / "sim" if data_root else None


def _exec(script: str, timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", CONTAINER, "bash", "-lc", f"python3.7 -c {script!r}"],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _container_up() -> bool:
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={CONTAINER}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return CONTAINER in result.stdout


def _mount_visible() -> bool:
    result = subprocess.run(
        ["docker", "exec", CONTAINER, "test", "-d", MOUNT], capture_output=True, check=False
    )
    return result.returncode == 0


def _town_opendrive(town: str, dest: Path) -> Path:
    """Copy the loaded town's OpenDRIVE out of the image so the real index can be built."""
    target = dest / f"{town}.xodr"
    subprocess.run(
        ["docker", "cp", f"{CONTAINER}:{OPENDRIVE_DIR}/{town}.xodr", str(target)],
        capture_output=True,
        check=True,
    )
    return target


def _current_town() -> str | None:
    """The town the server is already on. The scenario is built for this one.

    CARLA reports the loaded map as a content path and may name the streamed variant
    (``Town10HD_Opt``); the OpenDRIVE and the capability catalog both use the base name.
    """
    script = (
        "import carla;"
        "c = carla.Client('127.0.0.1', 2000); c.set_timeout(30.0);"
        "print('MAP', c.get_world().get_map().name.split('/')[-1])"
    )
    result = _exec(script, timeout=90)
    for line in result.stdout.splitlines():
        if line.startswith("MAP "):
            return line.split(None, 1)[1].strip().removesuffix("_Opt")
    return None


@pytest.fixture(scope="module")
def scenario_path(tmp_path_factory) -> str:
    """Compile one representative scenario into the mounted directory."""
    if not _container_up():
        pytest.skip(f"{CONTAINER} is not running — start the simulate profile")
    if not _mount_visible():
        pytest.skip(
            f"{MOUNT} is not mounted inside {CONTAINER}; recreate it so compose.simulate.yml "
            "picks up the artifacts bind mount"
        )
    host_dir = _host_dir()
    if host_dir is None:
        pytest.skip("neither SIM_ARTIFACT_DIR nor DATA_ROOT is set (.env not loaded)")
    town = _current_town()
    if town is None:
        pytest.skip("CARLA RPC did not answer — the server may still be starting")

    spec = SimSpec(
        clip_id="smoke-reconstruction",
        dna_version="0.2.0",
        duration_s=5.0,
        warmup_s=3.0,
        risk_level="critical",
        world=WorldSpec(
            weather={
                "cloudiness": 65.0,
                "precipitation": 30.0,
                "wetness": 40.0,
                "fog_density": 5.0,
                "sun_altitude_angle": 70.0,
                "sun_azimuth_angle": 250.0,
            },
            road=RoadQuery(
                road_type="secondary",
                intersection_type="signalized",
                min_driving_lanes=1,
                speed_kph_range=(40, 60),
                required_lane_types=("driving",),
                candidate_towns=(town,),
            ),
        ),
        ego=EgoSpec(
            maneuver="brake_hard", control_template="ego_brake_hard", target_speed_kph=45.0
        ),
        actors=(
            ActorSpec(
                index=0,
                actor_class="vehicle_car",
                blueprint_filter="vehicle.tesla.model3",
                state="cutin",
                maneuver_template="vehicle_lane_change_into_ego_lane",
                distance_bucket="near",
                distance_m=8.0,
                control_mode=ControlMode.EVENT,
            ),
        ),
        cameras=(CameraSpec(view="ego", transform=(1.6, 0.0, 1.5, 0.0, 0.0, 0.0)),),
        safety_event=SafetyEventSpec(
            has_event=True,
            event_type="near_miss",
            collision_type="none",
            severity_estimate="no_harm",
        ),
    )
    # The road must come from the town actually loaded: a hand-picked id is rejected by
    # the reader, which is precisely the class of error this stage exists to catch.
    xodr = _town_opendrive(town, tmp_path_factory.mktemp("xodr"))
    candidates = parse_town(xodr, town)
    road = select_road(spec.world.road, candidates, seed=spec.clip_id, min_length_m=100.0)
    assert road is not None, f"no road candidates parsed for {town}"

    root = compile_scenario(spec, road)
    assert validate(root, spec.clip_id).is_valid, "the document must pass our own XSD first"

    host_dir.mkdir(parents=True, exist_ok=True)
    (host_dir / SCENARIO_NAME).write_text(serialize(root), encoding="utf-8")
    return f"{MOUNT}/{SCENARIO_NAME}"


def test_scenario_runner_validator_accepts_the_document(scenario_path):
    """The pinned validator inside the image must agree with the host's newer one."""
    script = (
        "import xmlschema, os, srunner.scenarioconfigs.openscenario_configuration as m;"
        "xsd = os.path.join(os.path.dirname(os.path.abspath(m.__file__)),"
        " '../openscenario/OpenSCENARIO.xsd');"
        f"xmlschema.XMLSchema(xsd).validate({scenario_path!r});"
        "print('SRUNNER_XSD_OK')"
    )
    result = _exec(script)
    assert "SRUNNER_XSD_OK" in result.stdout, result.stderr[-2000:]


def test_carla_loads_the_scenario_world_and_entities(scenario_path):
    """The first point where anything CARLA-specific — road network, blueprints — can fail."""
    script = (
        "import carla;"
        "from srunner.scenarioconfigs.openscenario_configuration import"
        " OpenScenarioConfiguration;"
        "client = carla.Client('127.0.0.1', 2000); client.set_timeout(60.0);"
        f"cfg = OpenScenarioConfiguration({scenario_path!r}, client, {{}});"
        "print('TOWN', cfg.town);"
        "print('ENTITIES', len(cfg.other_actors) + len(cfg.ego_vehicles));"
        "print('CARLA_LOAD_OK')"
    )
    result = _exec(script, timeout=300)
    assert "CARLA_LOAD_OK" in result.stdout, result.stderr[-2000:]
    assert "ENTITIES 2" in result.stdout
