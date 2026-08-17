"""The corpus-wide acceptance criteria, enforced rather than measured once.

Compiling every curated segment is the actual deliverable, so "100% XSD-valid" and
"recompilation is byte-identical" are asserted against the real corpus and the real road
index, not against fixtures. Read-only: nothing here writes to the database.

Skips when Postgres is unreachable or the road index has not been built.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from my_curator.adapters.sim.xosc_writer import serialize, validate
from my_curator.adapters.storage.pg import PGRepository, dsn_from_env
from my_curator.domain.scout.versioning import CURRENT_DNA_VERSION
from my_curator.domain.sim import map_dna, select_road
from my_curator.domain.sim.road_index import RoadCandidate
from my_curator.domain.sim.xosc_compiler import KPH_TO_MPS, compile_scenario

pytestmark = pytest.mark.integration


def _as_candidate(row: dict) -> RoadCandidate:
    return RoadCandidate(
        town=row["town"],
        road_id=row["road_id"],
        lane_id=row["lane_id"],
        lane_section_s=row["lane_section_s"],
        lane_section_end_s=row["lane_section_end_s"],
        driving_lanes=row["driving_lanes"],
        speed_kph=row["speed_kph"],
        lane_types=frozenset(row["lane_types"]),
        junction_forms=frozenset(row["junction_forms"]),
        in_junction=row["in_junction"],
    )


def _dna_of(row: dict) -> dict:
    dna = row["dna_json"]
    if isinstance(dna, str):
        dna = json.loads(dna)
    dna.setdefault("clip_id", str(row["clip_id"]))
    return dna


async def _load() -> tuple[list[dict], list[dict]]:
    repo = await PGRepository.create(dsn_from_env())
    try:
        rows = await repo.list_dna(dna_version=CURRENT_DNA_VERSION, limit=5000)
        index = await repo.list_sim_road_index()
    finally:
        await repo.close()
    return rows, index


@pytest.fixture(scope="module")
def corpus():
    """Curated DNA plus the road index, straight from the deployed database."""
    if "PG_USER" not in os.environ:
        pytest.skip("Postgres environment not configured (.env not loaded)")
    try:
        rows, index = asyncio.run(_load())
    except Exception as exc:  # noqa: BLE001 — a connection failure is a skip, not a failure
        pytest.skip(f"Postgres unreachable: {exc}")
    if not rows:
        pytest.skip("no curated DNA in the database")
    if not index:
        pytest.skip("sim_road_index is empty — run my_curator.cli.build_road_index")
    return rows, [_as_candidate(r) for r in index]


def _compile_all(corpus) -> list[tuple]:
    rows, candidates = corpus
    out = []
    for row in rows:
        result = map_dna(_dna_of(row))
        if result.spec is None:
            continue
        spec = result.spec
        min_length = (spec.warmup_s + spec.duration_s) * spec.ego.target_speed_kph * KPH_TO_MPS
        selection = select_road(
            spec.world.road, candidates, seed=result.clip_id, min_length_m=min_length
        )
        out.append((result.clip_id, selection, compile_scenario(spec, selection)))
    return out


@pytest.fixture(scope="module")
def compiled(corpus):
    return _compile_all(corpus)


def test_every_mapped_segment_compiles(corpus, compiled):
    rows, _ = corpus
    mapped = sum(1 for row in rows if map_dna(_dna_of(row)).spec is not None)
    assert mapped > 0
    assert len(compiled) == mapped


def test_every_generated_scenario_is_schema_valid(compiled):
    invalid = [
        (clip_id, result.errors[:1])
        for clip_id, _, root in compiled
        if not (result := validate(root, clip_id)).is_valid
    ]
    assert not invalid, invalid[:3]


def test_every_road_query_resolves(compiled):
    assert all(selection is not None for _, selection, _ in compiled)


def test_recompilation_is_byte_identical(corpus, compiled):
    again = _compile_all(corpus)
    assert [serialize(root) for _, _, root in compiled] == [serialize(root) for _, _, root in again]


def test_every_scenario_names_the_town_it_was_staged_in(compiled):
    for _, selection, root in compiled:
        logic_file = root.find("RoadNetwork/LogicFile")
        assert logic_file is not None
        assert logic_file.get("filepath") == selection.candidate.town
