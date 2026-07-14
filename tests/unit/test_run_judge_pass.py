"""Unit tests for the run_judge_pass CLI arg parsing + scope selection (P4-6)."""

from __future__ import annotations

import argparse
import json
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from my_curator.cli.run_judge_pass import (
    DEFAULT_GOLD,
    _build_arg_parser,
    _select_records,
    attach_gt,
    load_gold_gt,
)

pytestmark = pytest.mark.unit

CLIP = "46c46784-a7f4-4809-92a9-61eacae8a2fc"


def _parse(argv):
    return _build_arg_parser().parse_args(argv)


def test_default_scope_has_no_session_or_all():
    a = _parse([])
    assert a.session is None and a.all_v02 is False and a.gold_set is None
    assert a.n_samples == 3 and a.dry_run is False


def test_gold_set_flag_uses_default_path_when_bare():
    assert _parse(["--gold-set"]).gold_set == DEFAULT_GOLD
    assert _parse(["--gold-set", "/tmp/g.json"]).gold_set == "/tmp/g.json"


def test_session_and_dry_run():
    a = _parse(["--session", "sess-1", "--dry-run"])
    assert a.session == "sess-1" and a.dry_run is True


def test_all_v02_flag():
    assert _parse(["--all-v0.2"]).all_v02 is True


def test_scope_mutually_exclusive():
    with pytest.raises(SystemExit):
        _parse(["--session", "s", "--all-v0.2"])


def test_load_gold_gt(tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(json.dumps({"clips": [{"clip_id": CLIP, "risk_level": "critical"}]}))
    assert load_gold_gt(str(p)) == {CLIP: "critical"}


def test_attach_gt_matches_by_str_clip_id():
    rows = [{"clip_id": UUID(CLIP), "dna_json": {}}]
    out = attach_gt(rows, {CLIP: "elevated"})
    assert out[0]["gt"] == "elevated"


def test_attach_gt_none_when_absent():
    rows = [{"clip_id": UUID(CLIP), "dna_json": {}}]
    assert attach_gt(rows, {})[0]["gt"] is None


async def test_select_records_session_scope():
    repo = AsyncMock()
    repo.list_v02_dna = AsyncMock(return_value=[{"clip_id": UUID(CLIP), "dna_json": {}}])
    rows = await _select_records(repo, _parse(["--session", "sess-1"]))
    repo.list_v02_dna.assert_awaited_once_with(session_id="sess-1", limit=1000)
    assert rows


async def test_select_records_all_v02_scope():
    repo = AsyncMock()
    repo.list_v02_dna = AsyncMock(return_value=[])
    await _select_records(repo, _parse(["--all-v0.2", "--limit", "50"]))
    repo.list_v02_dna.assert_awaited_once_with(limit=50)


async def test_select_records_gold_scope_attaches_gt(tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(json.dumps({"clips": [{"clip_id": CLIP, "risk_level": "critical"}]}))
    repo = AsyncMock()
    repo.list_v02_dna = AsyncMock(return_value=[{"clip_id": UUID(CLIP), "dna_json": {}}])
    rows = await _select_records(repo, _parse(["--gold-set", str(p)]))
    assert repo.list_v02_dna.await_args.kwargs["clip_ids"] == [UUID(CLIP)]
    assert rows[0]["gt"] == "critical"
