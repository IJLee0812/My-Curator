"""Bring the simulator up on a given town, and run the staging command inside it.

Map switching segfaults this build, so a town is chosen at boot and never changed: the
render tool owns the container's lifecycle and restarts it per town. Everything here is
``docker`` and ``docker compose`` driven as a subprocess, which is the only interface the
host has to a simulator it cannot import.
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
from pathlib import Path

log = logging.getLogger(__name__)

CONTAINER = "my-curator-carla"
SERVICE = "carla-server"
COMPOSE_FILES = ("infra/compose.base.yml", "infra/compose.simulate.yml")

BOOT_TIMEOUT_S = 180
_POLL_INTERVAL_S = 3.0

#: Camera capture needs the full renderer. At the compose default quality the server dies
#: with signal 11 the moment an RGB camera is asked for a frame, so a render always boots at
#: Epic regardless of what the environment says.
RENDER_QUALITY = "Epic"

#: The simulator answers its RPC port well before the world is usable; the readiness probe
#: asks the world for its map name, which is only possible once the town has loaded.
_READY_PROBE = "import carla; print(carla.Client('127.0.0.1', 2000).get_world().get_map().name)"


class SimulatorError(RuntimeError):
    """The simulator could not be brought into the state a render needs."""


def _compose(repo_root: Path, *args: str) -> list[str]:
    command = ["docker", "compose", "--env-file", str(repo_root / ".env")]
    for compose_file in COMPOSE_FILES:
        command += ["-f", str(repo_root / compose_file)]
    return command + ["--profile", "simulate", *args]


def _run(command: list[str], *, env: dict | None = None, timeout_s: int = 300) -> str:
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_s, env=env, check=False
    )
    if result.returncode != 0:
        raise SimulatorError(
            f"{command[0]} failed: {(result.stderr or result.stdout).strip()[:400]}"
        )
    return result.stdout


def _render_env(town: str) -> dict:
    return {"CARLA_MAP": town, "CARLA_QUALITY": RENDER_QUALITY}


def _port_open(host: str, port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1.0)
        return probe.connect_ex((host, port)) == 0


def loaded_town(container: str = CONTAINER) -> str | None:
    """The town the running simulator is on, or ``None`` if it is not answering yet."""
    result = subprocess.run(
        ["docker", "exec", container, "bash", "-lc", f'python3.7 -c "{_READY_PROBE}"'],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip().rsplit("/", 1)[-1] or None


def is_running(container: str = CONTAINER) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() == "true"


def shutdown(repo_root: Path) -> None:
    """Remove the simulator container, and only that.

    Never ``compose down``: the simulate profile is layered on ``compose.base.yml``, so a
    project-wide down takes Postgres, MinIO, Milvus and Kafka with it.
    """
    _run(_compose(repo_root, "rm", "--stop", "--force", SERVICE))


def boot(repo_root: Path, town: str, *, host: str, port: int, env: dict) -> None:
    """Start the simulator on *town*, restarting it only when the town has to change."""
    already_there = is_running() and loaded_town() == town
    if is_running() and not already_there:
        log.info("restarting simulator for %s", town)
        shutdown(repo_root)

    # Run compose even when the town already matches: it is a no-op unless the service
    # definition has moved on, and a container still holding an older set of mounts is
    # exactly the kind of drift that fails deep inside a render instead of here.
    _run(_compose(repo_root, "up", "-d", SERVICE), env={**env, **_render_env(town)})
    if already_there and loaded_town() == town:
        return

    deadline = time.time() + BOOT_TIMEOUT_S
    while time.time() < deadline:
        if _port_open(host, port):
            current = loaded_town()
            if current == town:
                log.info("simulator ready on %s", town)
                return
            if current is not None:
                raise SimulatorError(f"simulator booted on {current}, not {town}")
        time.sleep(_POLL_INTERVAL_S)
    raise SimulatorError(f"simulator did not become ready on {town} within {BOOT_TIMEOUT_S}s")


def run_stage(arguments: list[str], *, container: str = CONTAINER, timeout_s: int = 900) -> dict:
    """Execute the in-container staging command and return its JSON result."""
    command = ["docker", "exec", container, "python3.7", "-m", "my_curator.cli.run_sim_stage"]
    result = subprocess.run(
        command + arguments, capture_output=True, text=True, timeout=timeout_s, check=False
    )
    for line in reversed(result.stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    raise SimulatorError(
        f"staging produced no result: {(result.stderr or result.stdout).strip()[:400]}"
    )
