"""Prompt-hash → dna_version lookup table (P2-5).

PROMPT_VERSION_MAP maps the first 16 hex chars of a Scout prompt's SHA-256
to the dna_version string that should be stored in scenario_dna.dna_version.

Maintenance contract:
  - When a prompt file is changed, compute its new hash prefix and add an entry
    here before merging.  That is the *only* file that needs to change in this
    module between prompt versions.
  - dna_version values must match the `const` declared in
    schemas/scenario_dna_v0_1.schema.json.  Bumping the schema version requires
    a schema migration (§5.2 of implementation_plan.md) and a matching new entry
    in this map.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# hash_prefix (16 hex chars) → dna_version string
PROMPT_VERSION_MAP: dict[str, str] = {
    "f823defb040481ce": "0.1.0",  # prompts/scout_cosmos_reason2.v1.md (P2-4 baseline, pre-P2-6)
    "223892aa4a72b577": "0.1.0",  # prompts/scout_cosmos_reason2.v1.md (P2-6: DNA v0.1 CoT prompt)
}


def resolve_dna_version(prompt_hash: str) -> str:
    """Return the dna_version for *prompt_hash* (first 16 hex chars of SHA-256).

    Falls back to "0.1.0" with a warning when the hash is not registered so
    that the consumer never crashes on an unrecognised prompt.  Register the
    new hash in PROMPT_VERSION_MAP before deploying a prompt change.
    """
    version = PROMPT_VERSION_MAP.get(prompt_hash)
    if version is None:
        log.warning(
            "Prompt hash %r not in PROMPT_VERSION_MAP — defaulting dna_version to '0.1.0'. "
            "Register the hash before deploying this prompt.",
            prompt_hash,
        )
        return "0.1.0"
    return version


def assert_prompt_registered(prompt_hash: str) -> None:
    """Raise ValueError if *prompt_hash* is not in PROMPT_VERSION_MAP.

    Intended for CI / test startup assertions where an unregistered prompt is
    a hard error rather than a recoverable condition.
    """
    if prompt_hash not in PROMPT_VERSION_MAP:
        raise ValueError(
            f"Prompt hash {prompt_hash!r} is not registered in PROMPT_VERSION_MAP. "
            "Add an entry to src/scouts/versioning.py before merging this prompt change."
        )
