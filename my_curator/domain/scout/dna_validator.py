"""Post-aggregator JSON-Schema validator for Scenario DNA (P2-6; multi-version dispatch P4-1).

Importable without GStreamer, CUDA, or torch — safe for unit tests on the host.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import jsonschema.validators

_SCHEMA_DIR = Path(__file__).parent.parent.parent.parent / "schemas"

# dna_version -> schema filename. Every registered version is loaded and compiled
# once at construction; ``validate`` dispatches on the document's dna_version.
_SCHEMA_FILES: dict[str, str] = {
    "0.1.0": "scenario_dna_v0_1.schema.json",
    "0.2.0": "scenario_dna_v0_2.schema.json",
}
_CODE_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)

# Authored by the pipeline after generation (dna_normalizer.ensure_managed_fields).
# With additionalProperties:false, a decoder constrained by the raw schema would
# force the model to fabricate the whole envelope.
MANAGED_FIELDS: tuple[str, ...] = ("dna_version", "clip_id", "timestamp_range", "provenance")

# Generation-side only; the storage schema stays pattern-free so historical
# documents remain valid.  Per-sentence bounds make the grammar TERMINATING —
# without them a comma chain never emits a terminator and runs to max_tokens,
# losing the document.  Side effect: decimals ("0.5") are ungrammarable, so the
# prompt asks for whole numbers or words.
SCENE_DESCRIPTION_PATTERN = r"[^.!?]{1,199}[.!?] [^.!?]{1,199}[.!?] [^.!?]{1,199}[.!?]"
# One sentence, <= 300 chars (the stored budget).  Forcing the sentence to open by
# citing the deciding risk rule was measured and regressed recall sharply — the
# citation became boilerplate; do not re-add without a fresh A/B.
RISK_RATIONALE_PATTERN = r"[^.!?]{1,299}[.!?]"


def generation_schema(version: str = "0.2.0") -> dict[str, Any]:
    """The deployed schema minus the pipeline-managed envelope, for
    grammar-constrained decoding.  Everything else is kept, so a constrained
    generation followed by ``ensure_managed_fields`` validates against the
    full deployed schema.
    """
    filename = _SCHEMA_FILES.get(version)
    if filename is None:
        raise ValueError(f"unregistered dna_version: {version!r}")
    schema = json.loads((_SCHEMA_DIR / filename).read_text(encoding="utf-8"))
    for field in MANAGED_FIELDS:
        schema["properties"].pop(field, None)
    schema["required"] = [k for k in schema["required"] if k not in MANAGED_FIELDS]
    _drop_ungrammarable(schema)

    sd = schema["properties"].get("scene_description")
    if isinstance(sd, dict):
        sd.pop("maxLength", None)  # the pattern is the length constraint
        sd["pattern"] = SCENE_DESCRIPTION_PATTERN
    rationale = (
        schema["properties"]
        .get("planner_logic", {})
        .get("properties", {})
        .get("risk_level_rationale")
    )
    if isinstance(rationale, dict):
        rationale.pop("maxLength", None)  # pattern bounds it at 299 <= the stored 300
        rationale["pattern"] = RISK_RATIONALE_PATTERN
    return schema


def _drop_ungrammarable(obj: Any) -> None:
    """Strip constraints the decoder's schema whitelist rejects (currently only
    ``uniqueItems``); the deployed schema keeps them for post-generation validation."""
    if isinstance(obj, dict):
        obj.pop("uniqueItems", None)
        for value in obj.values():
            _drop_ungrammarable(value)
    elif isinstance(obj, list):
        for value in obj:
            _drop_ungrammarable(value)


class DNAValidator:
    """Post-aggregator JSON-Schema validator with dna_version-based dispatch.

    Loads every registered Scenario DNA schema once at construction and compiles
    one validator per version. ``validate`` dispatches on the document's
    ``dna_version``; an unknown or missing version is an explicit failure, never
    a silent fallback to v0.1.
    """

    def __init__(self) -> None:
        self._schemas: dict[str, dict[str, Any]] = {}
        self._validators: dict[str, Any] = {}
        for version, filename in _SCHEMA_FILES.items():
            raw = (_SCHEMA_DIR / filename).read_text(encoding="utf-8")
            schema = json.loads(raw)
            cls = jsonschema.validators.validator_for(schema)
            self._schemas[version] = schema
            self._validators[version] = cls(schema)

    def extract_json(self, text: str) -> dict[str, Any] | None:
        """3-stage extraction from CoT output.

        Stage 1: last ```json...``` code fence block.
        Stage 2: last outermost {...} balanced match.
        Stage 3: return None (caller routes to needs_review).
        """
        # Stage 1: last ```json...``` fence
        matches = _CODE_FENCE_RE.findall(text)
        if matches:
            candidate = matches[-1].strip()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass

        # Stage 2: last outermost {...} block
        extracted = _extract_last_object(text)
        if extracted is not None:
            return extracted

        # Stage 3: no parseable JSON found
        return None

    def validate(self, dna: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate *dna* against the schema for its ``dna_version``.

        Dispatches on ``dna["dna_version"]``. A missing or unregistered version
        is an explicit failure (no v0.1 fallback). Schemas are compiled once at
        __init__ — no repeated disk I/O.
        Returns (is_valid, [error_messages]).
        """
        version = dna.get("dna_version") if isinstance(dna, dict) else None
        validator = self._validators.get(version) if isinstance(version, str) else None
        if validator is None:
            return False, [
                f"unknown or missing dna_version: {version!r} "
                f"(registered versions: {sorted(self._validators)})"
            ]
        errors = sorted(validator.iter_errors(dna), key=lambda e: list(e.path))
        if errors:
            return False, [e.message for e in errors]
        return True, []


def _extract_last_object(text: str) -> dict[str, Any] | None:
    """Return the last outermost {...} block in *text* parsed as JSON, or None."""
    last_close = text.rfind("}")
    if last_close == -1:
        return None

    depth = 0
    for i in range(last_close, -1, -1):
        ch = text[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            depth -= 1
            if depth == 0:
                candidate = text[i : last_close + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, ValueError):
                    return None
    return None
