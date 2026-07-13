"""Judge prompt loading, hashing, and user-prompt construction (P4-6).

The judge prompt is versioned independently of the Scout prompt: a judge-prompt change
records a new ``judge_prompt_hash`` in ``scenario_dna.provenance`` but does **NOT** bump
``dna_version`` (the Judge is additive over v0.2 DNA). Hashing mirrors the Scout
convention — ``sha256(file_bytes)[:16]`` (see ``application/consumers/curation_consumer``).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"
JUDGE_PROMPT_FILE = "judge_qwen3.v1.md"

# Registered judge-prompt hashes (16 hex chars of sha256 over the file bytes).
# Add the new hash here when the judge prompt changes — the prompt_regression suite
# asserts the shipped file's hash is registered.
JUDGE_PROMPT_HASHES: set[str] = {
    "d06aef8a3365f0b2",  # prompts/judge_qwen3.v1.md (P4-6: SOTIF rubric + N-vote critic)
}


def _prompt_path(filename: str = JUDGE_PROMPT_FILE) -> Path:
    return _PROMPTS_DIR / filename


def judge_prompt_hash(filename: str = JUDGE_PROMPT_FILE) -> str:
    """Return the 16-hex-char sha256 prefix of the judge prompt file bytes."""
    return hashlib.sha256(_prompt_path(filename).read_bytes()).hexdigest()[:16]


def load_system_prompt(filename: str = JUDGE_PROMPT_FILE) -> str:
    """Return the judge system-prompt text (the file content, verbatim)."""
    return _prompt_path(filename).read_text(encoding="utf-8")


def assert_judge_prompt_registered(prompt_hash: str) -> None:
    """Raise ValueError if *prompt_hash* is not in JUDGE_PROMPT_HASHES."""
    if prompt_hash not in JUDGE_PROMPT_HASHES:
        raise ValueError(
            f"Judge prompt hash {prompt_hash!r} is not registered in JUDGE_PROMPT_HASHES. "
            "Add it to my_curator/domain/judge/prompt.py before merging the prompt change."
        )


def build_judge_user_prompt(dna: dict[str, Any]) -> str:
    """Render the four judged/context fields of a v0.2 DNA as the critic user message.

    Feeds ``scene_description`` + ``risk_level`` + ``risk_level_rationale`` +
    ``safety_event`` (read-only context); the enums are never re-fed separately.
    """
    pl = dna.get("planner_logic", {}) if isinstance(dna, dict) else {}
    se = pl.get("safety_event", {}) if isinstance(pl, dict) else {}
    return (
        "Scout Scenario DNA:\n"
        f'- scene_description: "{dna.get("scene_description", "")}"\n'
        f"- risk_level: {pl.get('risk_level')}\n"
        f'- risk_level_rationale: "{pl.get("risk_level_rationale", "")}"\n'
        f"- safety_event: has_event={se.get('has_event')}, event_type={se.get('event_type')}, "
        f"collision_type={se.get('collision_type')}, severity_estimate={se.get('severity_estimate')}\n\n"
        "Re-score risk_level and scene_description per your instructions."
    )
