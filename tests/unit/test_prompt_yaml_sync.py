"""Guard the Scout prompt's two-file split.

The prompt that runs lives in ``configs/config_driving_scene.yaml``, while the
hash stamped into ``provenance.scout_prompt_hash`` is SHA-256 of the markdown
artifact.  Nothing at runtime couples the two, so an edit to one alone would
record a provenance hash for a prompt that never ran.  These tests are that
coupling.  The markdown uses FOUR-backtick fences because the prompt itself
contains ```json exemplars.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest
import yaml

from my_curator.application.consumers.curation_consumer import _SCOUT_PROMPT_PATH
from my_curator.domain.scout.versioning import PROMPT_VERSION_MAP, resolve_dna_version

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs" / "config_driving_scene.yaml"
_FENCE = "````"

pytestmark = [pytest.mark.unit, pytest.mark.prompt_regression]


def _yaml_prompts() -> tuple[str, str]:
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    inference = cfg["inference"]
    return inference["system_prompt"], inference["user_prompt"]


def _fenced_block(markdown: str, heading: str) -> str:
    """Return the four-backtick fenced block that follows *heading*."""
    _, _, after_heading = markdown.partition(heading)
    assert after_heading, f"heading {heading!r} not found in the prompt artifact"
    lines = after_heading.splitlines(keepends=True)
    opening = next(i for i, line in enumerate(lines) if line.rstrip("\n") == _FENCE)
    body: list[str] = []
    for line in lines[opening + 1 :]:
        if line.rstrip("\n") == _FENCE:
            return "".join(body)
        body.append(line)
    raise AssertionError(f"unterminated {_FENCE} fence after {heading!r}")


class TestPromptArtifactMirrorsConfig:
    def test_system_prompt_matches_byte_for_byte(self):
        system_prompt, _ = _yaml_prompts()
        mirrored = _fenced_block(_SCOUT_PROMPT_PATH.read_text(encoding="utf-8"), "## System Prompt")
        assert mirrored == system_prompt, (
            f"{_SCOUT_PROMPT_PATH.name} has drifted from "
            "config_driving_scene.yaml inference.system_prompt — regenerate the "
            "artifact from the yaml so the recorded scout_prompt_hash describes "
            "the prompt that actually runs."
        )

    def test_user_prompt_matches_byte_for_byte(self):
        _, user_prompt = _yaml_prompts()
        mirrored = _fenced_block(
            _SCOUT_PROMPT_PATH.read_text(encoding="utf-8"), "## User Prompt Template"
        )
        assert mirrored == user_prompt, (
            f"{_SCOUT_PROMPT_PATH.name} has drifted from "
            "config_driving_scene.yaml inference.user_prompt."
        )


class TestLivePromptHashRegistered:
    def test_consumer_prompt_hash_resolves_to_v020(self):
        """The file the consumer hashes must be registered, else dna_version
        silently falls back to 0.1.0 and every row is mislabelled."""
        digest = hashlib.sha256(_SCOUT_PROMPT_PATH.read_bytes()).hexdigest()[:16]
        assert digest in PROMPT_VERSION_MAP, (
            f"{_SCOUT_PROMPT_PATH.name} hash {digest!r} is not in "
            "PROMPT_VERSION_MAP — register it before running the pipeline."
        )
        assert resolve_dna_version(digest) == "0.2.0"


class TestManagedEnvelopeNotRequested:
    """``ensure_managed_fields`` authors dna_version / clip_id / timestamp_range
    / provenance and runs before schema validation in both the publisher and the
    consumer, so asking the model for them buys nothing.  It also cost: a
    one-video probe (2026-08-14) had the model close
    ``provenance.reference_standards`` with ``}`` instead of ``]`` on every
    greedy segment, voiding the whole DNA object each time."""

    MANAGED = ("dna_version", "clip_id", "timestamp_range", "provenance")

    def test_exemplars_omit_the_managed_envelope(self):
        system_prompt, _ = _yaml_prompts()
        for key in self.MANAGED:
            assert f'"{key}"' not in system_prompt, (
                f"the prompt asks the model to emit {key!r}, which the pipeline "
                "overwrites — every token spent there is discarded and a "
                "malformed value there voids the entire DNA object."
            )

    def test_exemplars_are_still_valid_json(self):
        """Removing the envelope must not leave a trailing comma behind.

        Under structured decoding the response is fence-less (a JSON grammar
        makes ``` unreachable), so the exemplar is a bare JSON object under an
        ``## Exemplar`` heading — extract it by brace balancing, not fences."""
        import json

        system_prompt, _ = _yaml_prompts()
        _, _, after = system_prompt.partition("## Exemplar")
        assert after, "no '## Exemplar' heading in the system prompt"
        start = after.index("{")
        depth = 0
        for i, ch in enumerate(after[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    body = after[start : i + 1]
                    break
        else:
            raise AssertionError("unbalanced exemplar object")
        parsed = json.loads(body)  # raises on a dangling comma
        assert "planner_logic" in parsed
