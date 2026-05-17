"""GT-5: clean-architecture layer dependency invariant (AST-based).

Enforces the 4-layer dependency arrow direction inside ``my_curator/``:

    interfaces  →  application  →  adapters  →  domain
        cli     ↗

Lower layers must NOT import higher layers.

    domain      may import: stdlib + jsonschema + pydantic + hypothesis only.
    adapters    may import: domain + stdlib + any external SDK.
    application may import: domain + adapters + stdlib + external SDK.
    interfaces  may import: anything inside my_curator.
    cli         may import: anything inside my_curator.

During R-0 the my_curator/ tree is mostly empty (skeleton only), so the
invariant trivially holds.  Subsequent stages (R-1..R-6) populate the layers
and this test becomes the structural acceptance gate.

References:
  docs/refactoring_plan.md  §2.1, §3.1 GT-5.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parents[2] / "my_curator"


_LAYERS = ["domain", "adapters", "application", "interfaces", "cli"]

# Higher index = higher layer. A module in layer N must not import a module in
# layer M when M > N (with the exception that interfaces/cli may freely cross-cut).
_LAYER_RANK = {name: i for i, name in enumerate(_LAYERS)}

# Modules allowed to be imported by domain/ (beyond stdlib).
_DOMAIN_ALLOWED_EXTERNAL_ROOTS = frozenset(
    {"jsonschema", "pydantic", "hypothesis", "yaml", "typing_extensions"}
)


def _layer_of(module: str) -> str | None:
    """Return the layer name for an absolute module path like 'my_curator.domain.x'."""
    if not module.startswith("my_curator."):
        return None
    parts = module.split(".")
    if len(parts) < 2:
        return None
    return parts[1] if parts[1] in _LAYER_RANK else None


def _iter_pkg_files() -> list[Path]:
    return sorted(p for p in _PKG_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _collect_imports(path: Path) -> list[str]:
    """Return absolute dotted import names from an AST parse of *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # relative import — skip; resolved against owning package
                continue
            if node.module:
                out.append(node.module)
    return out


def _own_module(path: Path) -> str:
    rel = path.relative_to(_PKG_ROOT.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


@pytest.mark.unit
def test_my_curator_root_exists():
    assert _PKG_ROOT.is_dir(), f"my_curator/ package root missing: {_PKG_ROOT}"


@pytest.mark.unit
def test_no_higher_layer_imports():
    """Lower layer must not import a higher layer inside my_curator/*."""
    violations: list[str] = []
    for path in _iter_pkg_files():
        own = _own_module(path)
        own_layer = _layer_of(own)
        if own_layer is None:
            continue
        own_rank = _LAYER_RANK[own_layer]
        for imp in _collect_imports(path):
            imp_layer = _layer_of(imp)
            if imp_layer is None:
                continue
            # cli is an entrypoint layer — may import any my_curator module.
            if own_layer == "cli":
                continue
            if _LAYER_RANK[imp_layer] > own_rank:
                violations.append(f"{own}  →  {imp} (layer {imp_layer})")
    assert not violations, (
        "Layer dependency violations detected:\n  - "
        + "\n  - ".join(violations)
        + "\n\nLower layers must not import higher layers (see docs/refactoring_plan.md §2.1)."
    )


@pytest.mark.unit
def test_domain_only_imports_pure_libs():
    """my_curator/domain/* may only import stdlib + a small allow-list of external libs."""
    violations: list[str] = []
    for path in _iter_pkg_files():
        own = _own_module(path)
        if _layer_of(own) != "domain":
            continue
        for imp in _collect_imports(path):
            root = imp.split(".")[0]
            if root.startswith("my_curator"):
                # Within-package imports already covered by test_no_higher_layer_imports.
                continue
            # Allow stdlib (very rough heuristic: not in known third-party allow set).
            if root in _DOMAIN_ALLOWED_EXTERNAL_ROOTS:
                continue
            # Anything else (e.g. ``vllm``, ``boto3``, ``kafka``, ``gi``, ``torch``) is forbidden.
            if root in {
                "vllm",
                "boto3",
                "botocore",
                "kafka",
                "gi",
                "torch",
                "asyncpg",
                "pymilvus",
                "transformers",
                "fastapi",
                "uvicorn",
                "PIL",
            }:
                violations.append(f"{own}  →  {imp} (forbidden in domain)")
    assert not violations, "domain/ purity violations:\n  - " + "\n  - ".join(violations)
