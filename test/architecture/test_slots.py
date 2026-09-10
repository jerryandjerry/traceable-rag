"""Every slot is a real slot: a contract, a factory, and injection.

A slot is only swappable if three things hold. It publishes a Protocol, so a
replacement knows what to implement. It has a factory, so configuration picks
the implementation. And the pipeline obtains it through that factory rather
than importing one by name -- a single hardcoded import undoes the other two,
and nothing in the codebase notices.

Needs no database, no Elasticsearch and no network.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from test.architecture.layers import SLOTS, classify, slot_roots

SRC = Path(__file__).resolve().parents[2] / "backend" / "src" / "visionagent"

EXPECTED_SLOTS = frozenset({
    "intent", "planner", "executer", "rerank",
    "evaluator", "answer", "parsers", "vectorstore",
})

# Temporary contract exceptions. The empty default enforces every slot.
MISSING_CONTRACT: dict[str, str] = {}

# Temporary deep-import exceptions. The empty default enforces slot entry points.
DEEP_IMPORTS: dict[str, str] = {}

# Temporary state-construction exceptions. The empty default reserves this for pipeline/.
STATE_BUILT_ELSEWHERE: dict[str, str] = {}


def slot_dir(slot: str) -> Path:
    """Return the slot package under service/."""
    candidate = SRC / "service" / slot
    if candidate.is_dir():
        return candidate
    raise AssertionError(f"slot {slot!r} has no package")


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def pipeline_imports() -> list[tuple[str, int, str]]:
    out = []
    for path in sorted((SRC / "pipeline").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = str(path.relative_to(SRC.parent.parent))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                out.append((rel, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                out.extend((rel, node.lineno, a.name) for a in node.names)
    return [(f, n, m) for f, n, m in out if m.startswith("visionagent")]


def test_the_slot_list_is_exactly_what_is_claimed():
    """Eight slots, no more and no fewer, whatever the docs say."""
    assert SLOTS == EXPECTED_SLOTS


@pytest.mark.parametrize("slot", sorted(EXPECTED_SLOTS))
def test_every_slot_publishes_a_protocol_and_a_factory(slot):
    d = slot_dir(slot)
    missing = [f for f in ("base.py", "factory.py") if not (d / f).exists()]
    if slot in MISSING_CONTRACT:
        assert missing, (
            f"{slot} now has {('base.py', 'factory.py')} -- delete its "
            f"MISSING_CONTRACT entry ({MISSING_CONTRACT[slot]})"
        )
        return
    assert not missing, f"{slot} is not swappable: no {missing}"


def test_the_pipeline_reaches_no_slot_implementation():
    """Importing past a slot root is what makes its factory decorative."""
    roots = slot_roots()
    deep: list[str] = []
    for file, line, module in pipeline_imports():
        c = classify(module)
        if c is None or c[1] not in SLOTS:
            continue
        if module in roots[c[1]]:
            continue
        known = next((k for k in DEEP_IMPORTS if module == k or module.startswith(k + ".")), None)
        if known:
            continue
        deep.append(f"{file}:{line} -> {module} (slot root is {sorted(roots[c[1]])})")
    assert not deep, (
        "pipeline/ must enter a slot through its root:\n  " + "\n  ".join(deep)
    )


def test_the_known_deep_imports_still_exist():
    """A fixed deep import must leave DEEP_IMPORTS, or it stops guarding."""
    seen = {m for _, _, m in pipeline_imports()}
    stale = [
        k for k in DEEP_IMPORTS
        if not any(m == k or m.startswith(k + ".") for m in seen)
    ]
    assert not stale, f"fixed; delete from DEEP_IMPORTS: {stale}"


def test_the_query_pipeline_builds_every_slot_through_its_factory(monkeypatch):
    """Constructed with no arguments, it must call the factories, not import
    an implementation. Presence of the attribute proves nothing: an
    isinstance() check against a runtime_checkable Protocol passes for any
    object with the right attribute names.
    """
    import visionagent.pipeline.query as q

    called: list[str] = []
    for name in ("build_intent_parser", "build_planner", "build_evaluator",
                 "build_answer_generator", "build_executer", "build_reranker"):
        real = getattr(q, name)
        monkeypatch.setattr(
            q, name,
            (lambda *a, r=real, n=name, **kw: (called.append(n), r(*a, **kw))[1]),
        )
    q.QueryPipeline()
    assert sorted(called) == sorted([
        "build_intent_parser", "build_planner", "build_evaluator",
        "build_answer_generator", "build_executer", "build_reranker",
    ])


def test_the_query_pipeline_accepts_a_double_for_every_slot():
    """Injection has to work, or the factory is the only way and the slot is
    untestable in isolation."""
    import inspect

    from visionagent.pipeline.query import QueryPipeline

    params = inspect.signature(QueryPipeline.__init__).parameters
    for name in ("intent", "planner", "executer", "evaluator", "answer", "reranker"):
        assert name in params, f"cannot inject {name}"
        assert params[name].default is None, f"{name} is not optional"


def test_the_indexing_slot_offers_no_way_to_read():
    """A store that both writes and answers queries is why the retrieval tools
    imported this slot: they wanted the connection it happened to own.

    Keeping the read side off the contract is what makes that impossible to do
    again by accident, rather than a rule someone has to remember.
    """
    import ast

    src = (slot_dir("vectorstore") / "base.py").read_text(encoding="utf-8")
    proto = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.ClassDef) and n.name == "ChunkStore"
    )
    methods = {m.name for m in proto.body if isinstance(m, ast.FunctionDef)}
    assert not methods & {"search", "query", "get", "get_by_ids"}, (
        f"the indexing slot exposes a read method: {sorted(methods)}"
    )


def test_the_storage_layer_separates_reading_from_writing():
    """Read and write are separate Protocols so a caller can be handed one
    without the other. With one combined interface, the difference between a
    tool that can only search and a tool that could drop an index is a
    convention."""
    import ast

    src = (SRC / "database" / "base.py").read_text(encoding="utf-8")
    protos = {
        n.name: {m.name for m in n.body if isinstance(m, ast.FunctionDef)}
        for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef)
    }
    assert {"ChunkReader", "ChunkWriter", "GraphReader", "GraphWriter"} <= set(protos)

    mutating = {"index", "delete_document", "delete_index", "upsert", "save"}
    for name in ("ChunkReader", "GraphReader"):
        assert not protos[name] & mutating, (
            f"{name} exposes a mutation: {sorted(protos[name] & mutating)}"
        )


def test_orchestration_state_is_built_only_in_the_pipeline():
    """AgentState and ToolContext are the turn's state and identity. A layer
    that builds them is orchestrating, whatever its folder is called."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts or path.parts[-2:-1] == ("pipeline",):
            continue
        rel = str(path.relative_to(SRC.parent.parent))
        if rel in STATE_BUILT_ELSEWHERE or "/models/" in rel or "/tests/" in rel:
            continue
        src = path.read_text(encoding="utf-8")
        for name in ("AgentState(", "ToolContext("):
            if name in src and f"class {name[:-1]}" not in src:
                offenders.append(f"{rel} builds {name[:-1]}")
    assert not offenders, (
        "only pipeline/ may build orchestration state:\n  " + "\n  ".join(offenders)
    )
