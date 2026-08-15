"""Wiring checks for infra/compose.simulate.yml (P5-1).

Parses the compose file only — no Docker, no GPU.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
COMPOSE_PATH = REPO_ROOT / "infra" / "compose.simulate.yml"
ENTRYPOINT_PATH = REPO_ROOT / "infra" / "carla-entrypoint.sh"
DOCKERFILE_PATH = REPO_ROOT / "infra" / "carla.Dockerfile"


@pytest.fixture(scope="module")
def service() -> dict:
    data = yaml.safe_load(COMPOSE_PATH.read_text())
    return data["services"]["carla-server"]


@pytest.mark.unit
def test_files_exist():
    assert COMPOSE_PATH.exists()
    assert ENTRYPOINT_PATH.exists()
    assert DOCKERFILE_PATH.exists()


@pytest.mark.unit
def test_gated_behind_simulate_profile(service):
    assert service["profiles"] == ["simulate"]


@pytest.mark.unit
def test_image_tag_pinned(service):
    tag = service["image"].rsplit(":", 1)[1]
    assert tag not in {"latest", "main", "master"}


@pytest.mark.unit
def test_pinned_to_gpu_zero(service):
    assert service["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    nvidia = [d for d in devices if d.get("driver") == "nvidia"]
    assert nvidia, "carla-server must reserve an nvidia device"
    assert nvidia[0]["device_ids"] == ["0"], "GPU 1 is reserved for the DeepStream pipeline"


@pytest.mark.unit
def test_not_restarted_by_the_daemon(service):
    """An on-demand profile must not be resurrected holding a GPU."""
    assert str(service["restart"]) == "no"


@pytest.mark.unit
def test_publishes_rpc_and_viewer_ports(service):
    ports = [str(p) for p in service["ports"]]
    assert "network_mode" not in service

    rpc = next(p for p in ports if p.endswith(":2000-2002"))
    host_default = rpc.rsplit(":", 1)[0].split(":-")[-1].rstrip("}")
    start, end = host_default.split("-")
    assert int(end) - int(start) == 2, "host RPC range must preserve the container offsets"

    assert any(p.endswith(":6080") for p in ports)


@pytest.mark.unit
def test_joins_the_curation_network(service):
    assert service["networks"] == ["curation-net"]


@pytest.mark.unit
def test_entrypoint_is_bind_mounted(service):
    assert any("carla-entrypoint.sh:/carla-entrypoint.sh:ro" in v for v in service["volumes"])
    assert service["entrypoint"][-1] == "/carla-entrypoint.sh"


@pytest.mark.unit
def test_vulkan_icd_mounted_read_only(service):
    assert any(
        v.startswith("/usr/share/vulkan/icd.d:") and v.endswith(":ro") for v in service["volumes"]
    )


@pytest.mark.unit
def test_entrypoint_supports_both_render_modes():
    body = ENTRYPOINT_PATH.read_text()
    assert "CARLA_VIEWER" in body
    assert "-RenderOffScreen" in body
    assert "websockify" in body
