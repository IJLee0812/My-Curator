"""Integration probes for infra/compose.base.yml (P1-2).

These tests verify the compose file is structurally sound and — when Docker
is available and the stack is already running — that the three services are
healthy, the five MinIO buckets exist, the Postgres schema is initialized,
and the Milvus health endpoint responds.

The tests that require a running stack are auto-skipped unless the operator
brought the stack up ahead of time:
    docker compose -f infra/compose.base.yml --env-file .env up -d
This keeps the normal `pytest -m integration` run fast on machines without
Docker, while still giving a real gate for PR reviewers who do have it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import socket
import subprocess
import urllib.request

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).parents[2]
COMPOSE_PATH = REPO_ROOT / "infra" / "compose.base.yml"
COMPOSE_CURATE_PATH = REPO_ROOT / "infra" / "compose.curate.yml"
INIT_SQL_PATH = REPO_ROOT / "infra" / "init-sql" / "001_schema.sql"
INIT_SQL_004_PATH = REPO_ROOT / "infra" / "init-sql" / "004_source_clip_id.sql"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"

EXPECTED_BUCKETS = {"raw", "clips", "frames", "artifacts", "milvus"}
EXPECTED_SERVICES = {"minio", "minio-init", "postgres", "milvus-etcd", "milvus"}


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


DOCKER_AVAILABLE = _docker_available()
STACK_UP = DOCKER_AVAILABLE and _tcp_open("127.0.0.1", 9000)

requires_docker = pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker daemon not reachable")
requires_stack_up = pytest.mark.skipif(
    not STACK_UP,
    reason="Storage stack not up (run: docker compose -f infra/compose.base.yml --env-file .env up -d)",
)


# ── Structural checks — run without Docker ───────────────────────────────────


@pytest.mark.integration
def test_compose_file_parses():
    data = yaml.safe_load(COMPOSE_PATH.read_text())
    assert data["name"] == "my-curator-base"
    assert set(data["services"].keys()) == EXPECTED_SERVICES


@pytest.mark.integration
def test_compose_pins_every_image_tag():
    """No floating 'latest' / bare 'postgres:17' tags — all images pinned."""
    data = yaml.safe_load(COMPOSE_PATH.read_text())
    for name, spec in data["services"].items():
        image = spec["image"]
        assert ":" in image, f"{name}: image must be tagged, got {image!r}"
        tag = image.rsplit(":", 1)[1]
        assert tag not in {"latest", "main", "master"}, f"{name}: floating tag {tag!r} not allowed"


@pytest.mark.integration
def test_milvus_pinned_to_gpu_zero():
    data = yaml.safe_load(COMPOSE_PATH.read_text())
    devices = data["services"]["milvus"]["deploy"]["resources"]["reservations"]["devices"]
    nvidia = [d for d in devices if d.get("driver") == "nvidia"]
    assert nvidia, "milvus must reserve at least one nvidia device"
    assert nvidia[0]["device_ids"] == ["0"], (
        "GPU 1 is reserved for the DeepStream pipeline — Milvus must pin to GPU 0"
    )


@pytest.mark.integration
def test_compose_uses_data_root_env():
    """Every bind mount under ${DATA_ROOT} — no hardcoded NAS paths."""
    raw = COMPOSE_PATH.read_text()
    for sensitive in ("/k8s_volume_demo", "/mnt/nas", "/nas"):
        assert sensitive not in raw, f"hardcoded path {sensitive!r} leaked into compose"
    assert "${DATA_ROOT}" in raw


@pytest.mark.integration
def test_init_sql_creates_expected_tables():
    sql = INIT_SQL_PATH.read_text()
    for table in ("sessions", "clips", "scenario_dna", "review_queue"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
    assert "USING GIN" in sql and "dna_json" in sql


@pytest.mark.integration
def test_env_example_declares_required_vars():
    body = ENV_EXAMPLE_PATH.read_text()
    for var in ("DATA_ROOT", "MINIO_USER", "MINIO_PASSWORD", "PG_USER", "PG_PASSWORD"):
        assert f"{var}=" in body, f".env.example missing {var}"


# ── Live-stack probes — skipped unless the operator already brought it up ──


@requires_stack_up
@pytest.mark.integration
def test_minio_buckets_present():
    """All five buckets were created by the minio-init sidecar."""
    r = subprocess.run(
        [
            "docker",
            "exec",
            "my-curator-minio",
            "mc",
            "alias",
            "set",
            "local",
            "http://localhost:9000",
            os.environ.get("MINIO_USER", "minio-admin"),
            os.environ.get("MINIO_PASSWORD", "change-me-please"),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode == 0, r.stderr

    r = subprocess.run(
        ["docker", "exec", "my-curator-minio", "mc", "ls", "local"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode == 0, r.stderr
    for bucket in EXPECTED_BUCKETS:
        assert bucket in r.stdout, f"bucket {bucket!r} missing: {r.stdout}"


@requires_stack_up
@pytest.mark.integration
def test_postgres_schema_initialized():
    """sessions/clips/scenario_dna/review_queue exist + GIN index on dna_json."""
    user = os.environ.get("PG_USER", "curation")
    r = subprocess.run(
        [
            "docker",
            "exec",
            "my-curator-postgres",
            "psql",
            "-U",
            user,
            "-d",
            "curation",
            "-tAc",
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' ORDER BY table_name;",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert r.returncode == 0, r.stderr
    tables = {line.strip() for line in r.stdout.splitlines() if line.strip()}
    assert {"sessions", "clips", "scenario_dna", "review_queue"}.issubset(tables)

    r = subprocess.run(
        [
            "docker",
            "exec",
            "my-curator-postgres",
            "psql",
            "-U",
            user,
            "-d",
            "curation",
            "-tAc",
            "SELECT indexname FROM pg_indexes "
            "WHERE tablename='scenario_dna' AND indexdef LIKE '%gin%';",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "idx_scenario_dna_json_gin" in r.stdout


@requires_stack_up
@pytest.mark.integration
def test_milvus_healthz():
    with urllib.request.urlopen("http://127.0.0.1:9091/healthz", timeout=5) as resp:
        assert resp.status == 200


# ── P3-4: compose.curate.yml ui service + init-sql 004 ──────────────────────


@pytest.mark.integration
def test_init_sql_004_adds_source_clip_id_column():
    """004_source_clip_id.sql introduces clips.source_clip_id (P3-4)."""
    sql = INIT_SQL_004_PATH.read_text()
    assert "ADD COLUMN IF NOT EXISTS source_clip_id" in sql
    assert "idx_clips_source_clip_id" in sql


@pytest.mark.integration
def test_compose_curate_declares_ui_service():
    """compose.curate.yml exposes the Next.js UI on host port 3000 (P3-4)."""
    data = yaml.safe_load(COMPOSE_CURATE_PATH.read_text())
    assert "ui" in data["services"], "compose.curate.yml is missing the 'ui' service"
    ui = data["services"]["ui"]
    assert "build" in ui, "ui service must define a build context"
    ports = [str(p) for p in ui.get("ports", [])]
    assert any("3000:3000" in p for p in ports), f"ui port mapping missing: {ports}"
    depends = ui.get("depends_on", {})
    # depends_on may be a list or a dict in compose v2 — accept both shapes
    if isinstance(depends, dict):
        assert "curation-api" in depends
    else:
        assert "curation-api" in depends
