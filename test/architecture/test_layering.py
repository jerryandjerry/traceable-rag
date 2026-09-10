"""The one-way dependency, enforced.

    api/                    HTTP: auth, job issuance, wire encoding
      pipeline/             the order of the steps, and the run's state
        service/<slot>/     one step each; peers, never importing each other
          database/         connections and persisted formats
            providers/      LLM, embedding, web search, rerank clients
              utils/        stateless helpers
              models/ config/ exceptions/     pure data

A layer may import ANY layer below it, at any distance -- service/ reaches
utils/ or models/ directly, without passing through database/. What it may
never do is import upward, or sideways into a peer slot.

The test walks every first-party module, resolves each import to a layer, and
fails on anything that points sideways into a peer slot or upward into a caller.

Needs no database, no Elasticsearch and no network.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from test.architecture.layers import classify, is_violation

SRC = Path(__file__).resolve().parents[2] / "backend" / "src" / "visionagent"

def module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imports_of(path: Path) -> list[tuple[int, str]]:
    """Every first-party module this file imports, with its line number."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    found: list[tuple[int, str]] = []
    package = module_name(path).rsplit(".", 1)[0] if path.name != "__init__.py" else module_name(path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.append((node.lineno, node.module))
            else:
                # Relative and absolute imports represent the same dependency edge.
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                target = ".".join(base + ([node.module] if node.module else []))
                found.append((node.lineno, target))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, a.name) for a in node.names)
    return [(n, m) for n, m in found if m.startswith("visionagent")]


def current_violations() -> list[str]:
    """Every edge that breaks the rule, as stable `path:line src -> dst` lines."""
    out: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        src = module_name(path)
        rel = path.relative_to(SRC.parent.parent)
        for line, dst in imports_of(path):
            if is_violation(src, dst):
                out.append(f"{rel}:{line} {src} -> {dst}")
    return sorted(out)


def test_nothing_imports_sideways_or_upward():
    """The whole rule, in one assertion."""
    bad = current_violations()
    assert not bad, (
        "these imports break the one-way dependency:\n  " + "\n  ".join(bad)
    )


def test_api_cannot_bypass_pipeline_into_any_service_unit():
    """The rule covers future support units, not only today's slot names."""
    assert is_violation(
        "visionagent.api.example", "visionagent.service.runtime"
    ) is not None
    assert is_violation(
        "visionagent.api.example", "visionagent.service.future_slot"
    ) is not None


def test_the_api_never_touches_storage_directly():
    """api/ reads through typed repository methods and runs workflows; it
    does not hold a connection, a table, raw SQL or a search client.

    A text scan rather than an import walk, because a `from sqlalchemy import
    text` inside a function body is the same violation as one at the top.
    """
    from test.architecture.layers import API_STORAGE_MARKERS

    offenders: list[str] = []
    for path in sorted((SRC / "api").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for marker in API_STORAGE_MARKERS:
                if marker in code:
                    offenders.append(
                        f"{path.relative_to(SRC.parent.parent)}:{n} {marker!r}"
                    )
    assert not offenders, (
        "api/ reaches storage directly; move the write into a pipeline or the "
        "read into a repository method:\n  " + "\n  ".join(offenders)
    )


def test_the_api_calls_no_repository_write():
    """The repository exemption covers reads. A route that creates or deletes
    through a repository is running a workflow from the HTTP layer."""
    from test.architecture.layers import API_REPOSITORY_WRITE

    offenders: list[str] = []
    for path in sorted((SRC / "api").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if API_REPOSITORY_WRITE.search(line.split("#", 1)[0]):
                offenders.append(f"{path.relative_to(SRC.parent.parent)}:{n} {line.strip()}")
    assert not offenders, "api/ writes through a repository:\n  " + "\n  ".join(offenders)


def test_upstream_vendor_code_never_imports_first_party():
    """Only the two project-owned shims may reach into this package."""
    from test.architecture.layers import VENDOR_SHIMS

    offenders: list[str] = []
    for path in sorted((SRC / "vendor").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name = module_name(path)
        if name in VENDOR_SHIMS:
            continue
        for line, dst in imports_of(path):
            c = classify(dst)
            if c is None or c[1] == "vendor":   # vendored code calling itself
                continue
            offenders.append(f"{path.relative_to(SRC.parent.parent)}:{line} -> {dst}")
    assert not offenders, (
        "vendored upstream code must stay independent of this package; only "
        f"{sorted(VENDOR_SHIMS)} may bridge:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("leaf", ["models", "config", "exceptions"])
def test_the_leaves_import_nothing_above_them(leaf):
    """A data leaf that imports a service is no longer a leaf."""
    bad: list[str] = []
    for path in sorted((SRC / leaf).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        src = module_name(path)
        for line, dst in imports_of(path):
            if is_violation(src, dst):
                bad.append(f"{path.relative_to(SRC.parent.parent)}:{line} -> {dst}")
    assert not bad, f"{leaf}/ must stay a pure data leaf:\n  " + "\n  ".join(bad)
