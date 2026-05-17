"""Post-aggregator JSON-Schema validator for Scenario DNA v0.1 (P2-6).

Importable without GStreamer, CUDA, or torch — safe for unit tests on the host.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import jsonschema
import jsonschema.validators

_SCHEMA_PATH = (
    Path(__file__).parent.parent.parent.parent / "schemas" / "scenario_dna_v0_1.schema.json"
)
_CODE_FENCE_RE = re.compile(r"```json\s*(.*?)```", re.DOTALL)


class DNAValidator:
    """Post-aggregator JSON-Schema validator for Scenario DNA v0.1."""

    def __init__(self) -> None:
        raw = _SCHEMA_PATH.read_text(encoding="utf-8")
        self._schema: dict[str, Any] = json.loads(raw)

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
        """Validate *dna* against scenario_dna_v0_1.schema.json.

        Schema is loaded once at __init__ — no repeated disk I/O.
        Returns (is_valid, [error_messages]).
        """
        cls = jsonschema.validators.validator_for(self._schema)
        validator = cls(self._schema)
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
