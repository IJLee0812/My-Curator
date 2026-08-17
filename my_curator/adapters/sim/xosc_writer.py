"""Serialize and XSD-validate compiled OpenSCENARIO documents.

Validation lives here rather than in ``domain/`` because it needs ``xmlschema``, which the
layer rules keep out of the pure layer. It needs no container and no simulator, so the
whole curated corpus can be checked in CI.

The vendored schema is ASAM OpenSCENARIO V1.0.0, redistributed under ASAM's own terms as
stated in the file header — the same way CARLA's ``scenario_runner`` ships it.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import xmlschema

log = logging.getLogger(__name__)

XSD_PATH = Path(__file__).resolve().parents[3] / "schemas" / "OpenSCENARIO_1.0.xsd"

_XML_DECLARATION = '<?xml version="1.0" encoding="UTF-8"?>\n'


@dataclass(frozen=True)
class ValidationResult:
    clip_id: str
    is_valid: bool
    errors: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.is_valid


@lru_cache(maxsize=1)
def _schema() -> xmlschema.XMLSchema:
    """Compile the XSD once; it costs about a second and is reused across a corpus run."""
    return xmlschema.XMLSchema(str(XSD_PATH))


def serialize(root: ET.Element) -> str:
    """Render the document. Indented in place, so the same tree always yields the same text."""
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return _XML_DECLARATION + body + "\n"


def validate(root: ET.Element, clip_id: str = "") -> ValidationResult:
    """Check one document against the vendored schema, collecting every error found."""
    errors = tuple(str(e) for e in _schema().iter_errors(root))
    return ValidationResult(clip_id=clip_id, is_valid=not errors, errors=errors)


def write(root: ET.Element, path: Path) -> Path:
    """Write the document, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize(root), encoding="utf-8")
    return path
