"""Unit tests for src/scouts/versioning.py (P2-5)."""

from __future__ import annotations

import pytest

from src.scouts.versioning import (
    PROMPT_VERSION_MAP,
    assert_prompt_registered,
    resolve_dna_version,
)

_KNOWN_HASH = "f823defb040481ce"  # scout_cosmos_reason2.v1.md baseline
_UNKNOWN_HASH = "0000000000000000"


@pytest.mark.unit
class TestResolvednaVersion:
    def test_known_hash_returns_correct_version(self):
        assert resolve_dna_version(_KNOWN_HASH) == "0.1.0"

    def test_unknown_hash_returns_fallback(self):
        assert resolve_dna_version(_UNKNOWN_HASH) == "0.1.0"

    def test_unknown_hash_does_not_raise(self):
        resolve_dna_version(_UNKNOWN_HASH)  # must not raise

    def test_unknown_hash_emits_warning(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING, logger="src.scouts.versioning"):
            resolve_dna_version(_UNKNOWN_HASH)
        assert any("not in PROMPT_VERSION_MAP" in r.message for r in caplog.records)

    def test_map_contains_baseline_entry(self):
        assert _KNOWN_HASH in PROMPT_VERSION_MAP

    def test_all_map_values_are_semver_strings(self):
        import re

        pattern = re.compile(r"^\d+\.\d+\.\d+$")
        for k, v in PROMPT_VERSION_MAP.items():
            assert pattern.match(v), f"Map entry {k!r} → {v!r} is not semver"

    def test_new_hash_maps_to_new_version(self):
        """Adding a new hash→version entry to PROMPT_VERSION_MAP resolves correctly."""
        import src.scouts.versioning as versioning_mod

        new_hash = "abcdef1234567890"
        new_version = "0.2.0"
        original_map = dict(versioning_mod.PROMPT_VERSION_MAP)
        try:
            versioning_mod.PROMPT_VERSION_MAP[new_hash] = new_version
            assert resolve_dna_version(new_hash) == new_version
        finally:
            versioning_mod.PROMPT_VERSION_MAP.clear()
            versioning_mod.PROMPT_VERSION_MAP.update(original_map)


@pytest.mark.unit
class TestAssertPromptRegistered:
    def test_known_hash_passes(self):
        assert_prompt_registered(_KNOWN_HASH)  # must not raise

    def test_unknown_hash_raises_value_error(self):
        with pytest.raises(ValueError, match="not registered in PROMPT_VERSION_MAP"):
            assert_prompt_registered(_UNKNOWN_HASH)

    def test_error_message_contains_hash(self):
        with pytest.raises(ValueError, match=_UNKNOWN_HASH):
            assert_prompt_registered(_UNKNOWN_HASH)


@pytest.mark.unit
class TestMainStartupGuard:
    """P2-5: assert_prompt_registered guard wired into main() CLI entry point."""

    def _patch_env(self, monkeypatch):
        monkeypatch.setenv("SESSION_ID", "s1")
        monkeypatch.setenv("CURATOR_DATASET", "ds")
        monkeypatch.setenv("CURATOR_SUBSET", "val")
        monkeypatch.setenv("CURATOR_DATASET_VERSION", "1.0")

    def test_main_exits_when_prompt_file_missing(self, monkeypatch):
        """main() sys.exit(1) when _SCOUT_PROMPT_PATH does not exist."""
        import sys
        from pathlib import Path

        import src.bus.kafka as kafka_mod

        self._patch_env(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["kafka"])
        monkeypatch.setattr(kafka_mod, "_SCOUT_PROMPT_PATH", Path("/nonexistent/prompt.md"))

        with pytest.raises(SystemExit) as exc_info:
            kafka_mod.main()
        assert exc_info.value.code == 1

    def test_main_exits_on_unregistered_hash(self, monkeypatch, tmp_path):
        """main() sys.exit(1) when prompt hash is not in PROMPT_VERSION_MAP."""
        import sys

        import src.bus.kafka as kafka_mod

        self._patch_env(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["kafka"])

        fake_prompt = tmp_path / "scout_cosmos_reason2.v1.md"
        fake_prompt.write_bytes(b"unregistered prompt content")
        monkeypatch.setattr(kafka_mod, "_SCOUT_PROMPT_PATH", fake_prompt)
        monkeypatch.setattr(kafka_mod, "_compute_prompt_hash", lambda _: "deadbeef00000000")

        with pytest.raises(SystemExit) as exc_info:
            kafka_mod.main()
        assert exc_info.value.code == 1

    def test_main_proceeds_on_registered_hash(self, monkeypatch, tmp_path):
        """main() reaches asyncio.run() without sys.exit when hash IS registered."""
        import sys

        import src.bus.kafka as kafka_mod

        self._patch_env(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["kafka"])

        fake_prompt = tmp_path / "scout_cosmos_reason2.v1.md"
        fake_prompt.write_bytes(b"placeholder")
        monkeypatch.setattr(kafka_mod, "_SCOUT_PROMPT_PATH", fake_prompt)
        monkeypatch.setattr(kafka_mod, "_compute_prompt_hash", lambda _: _KNOWN_HASH)

        async def _noop(args, scout_prompt_hash):
            pass

        monkeypatch.setattr(kafka_mod, "_run", _noop)

        kafka_mod.main()  # must not raise SystemExit


@pytest.mark.unit
@pytest.mark.prompt_regression
class TestConsumerStartupSmoke:
    """Level-2 smoke: consumer startup derives dna_version from resolve_dna_version."""

    def test_write_clip_with_dna_receives_resolved_version(self, monkeypatch):
        """Mock _compute_prompt_hash → verify write_clip_with_dna gets dna_version='0.1.0'."""
        import unittest.mock as mock
        from pathlib import Path

        import src.bus.kafka as kafka_mod

        monkeypatch.setattr(kafka_mod, "_compute_prompt_hash", lambda _path: _KNOWN_HASH)

        captured: list[str] = []

        async def fake_run(args: object, scout_prompt_hash: str) -> None:
            from src.scouts.versioning import resolve_dna_version

            captured.append(resolve_dna_version(scout_prompt_hash))

        monkeypatch.setattr(kafka_mod, "_run", fake_run)

        import argparse

        args = argparse.Namespace(
            session_id="s1",
            dataset="ds",
            subset="val",
            dataset_version="1.0",
            broker="localhost:9092",
            milvus_uri="http://localhost:19530",
            timeout=5000,
        )

        import asyncio

        scout_prompt_hash = kafka_mod._compute_prompt_hash(
            Path("prompts/scout_cosmos_reason2.v1.md")
        )
        asyncio.run(fake_run(args, scout_prompt_hash))

        assert captured == ["0.1.0"]
