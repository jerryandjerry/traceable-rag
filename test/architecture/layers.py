"""Where every module sits in the one-way dependency, and what it may import.

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

Two slots that import each other are one slot wearing two names, and neither
can be replaced without the other, which is why sideways is refused as firmly
as upward. A module may always import anything inside its own unit.

Modules are mapped by dotted path so dependency edges are classified at import
boundaries rather than inferred from filesystem traversal.
"""
from __future__ import annotations

import re

API, PIPELINE, SERVICE, DATABASE, PROVIDERS, UTILS, LEAF, VENDOR = range(8)

LAYER_NAMES = {
    API: "api", PIPELINE: "pipeline", SERVICE: "service", PROVIDERS: "providers",
    DATABASE: "database", UTILS: "utils", LEAF: "leaf", VENDOR: "vendor",
}

# Longest prefix wins. `unit` separates peer slots: two entries sharing a layer
# but not a unit may not import each other.
#
# (module prefix, layer, unit)
MODULE_LAYERS: list[tuple[str, int, str]] = [
    ("visionagent.api", API, "api"),

    ("visionagent.pipeline", PIPELINE, "pipeline"),
    # Account workflows coordinate authentication and multiple stores.
    ("visionagent.pipeline.account", PIPELINE, "pipeline"),

    # The eight slots. Each is its own unit; tools/ is the executer's, so the
    # two share a unit and may import each other.
    ("visionagent.service.intent", SERVICE, "intent"),
    ("visionagent.service.planner", SERVICE, "planner"),
    ("visionagent.service.executer", SERVICE, "executer"),
    ("visionagent.service.executer.tools", SERVICE, "executer"),
    ("visionagent.service.evaluator", SERVICE, "evaluator"),
    ("visionagent.service.answer", SERVICE, "answer"),
    ("visionagent.service.parsers", SERVICE, "parsers"),
    ("visionagent.service.vectorstore", SERVICE, "vectorstore"),
    # Cross-slot provider lifecycle, not a configurable execution slot. The
    # leading underscore excludes this support unit from SLOTS below.
    ("visionagent.service.runtime", SERVICE, "_runtime"),
    # The reranker client is an external I/O adapter; the slot around it is not.
    ("visionagent.providers.rerank.dashscope.provider", PROVIDERS, "providers"),
    ("visionagent.service.rerank", SERVICE, "rerank"),
    # Classify future service packages too. Longest-prefix matching keeps the
    # known slots above in their own peer units.
    ("visionagent.service", SERVICE, "_service"),

    # Stateful I/O adapters: model, embedding, search and rerank clients.
    ("visionagent.providers", PROVIDERS, "providers"),

    # Connections and persisted formats.
    ("visionagent.database", DATABASE, "database"),
    ("visionagent.database.session_context", DATABASE, "database"),

    # Stateless helpers.
    ("visionagent.utils.keyword_extraction", UTILS, "utils"),
    ("visionagent.utils", UTILS, "utils"),

    ("visionagent.models", LEAF, "leaf"),
    ("visionagent.config", LEAF, "leaf"),
    ("visionagent.exceptions", LEAF, "leaf"),

    ("visionagent.vendor", VENDOR, "vendor"),
]

# The only files inside vendor/ that reach first-party code. They are
# project-owned shims, not upstream RAGFlow: model.py gives the vendored engine
# this system's embedder and reranker, and search_v2.py was edited to log
# through this package. Each may reach the units listed and nothing else; every
# other vendored file must reach nothing first-party at all.
VENDOR_SHIMS: dict[str, set[str]] = {
    "visionagent.vendor.ragflow.rag.nlp.model": {"providers"},
    "visionagent.vendor.ragflow.rag.nlp.search_v2": {"utils"},
}


# Downward edges the approved matrix still refuses. The total order says api/
# may import anything below it, but api/ is HTTP: it may run a workflow
# (pipeline) and read through typed repository methods, and nothing else --
# not a provider, not a vendored engine, not a table or a connection. pipeline/
# orchestrates slots; a pipeline that reaches a provider or vendored code
# directly has bypassed the slot that owns it.
#
# unit -> (forbidden destination units, module prefixes exempt from the ban)
FORBIDDEN_DOWNWARD: dict[str, tuple[frozenset[str], tuple[str, ...]]] = {
    "api": (
        frozenset({
            "providers", "vendor", "database",
        }),
        ("visionagent.database.postgres.repositories",),
    ),
    "pipeline": (frozenset({"providers", "vendor"}), ()),
}

# The repository module exposes writes as well as reads. api/ may call the
# reads; a write from a route is a workflow that belongs in pipeline/.
API_REPOSITORY_WRITE = re.compile(r"Repository\(\)\s*\.\s*(create|delete|save|insert|update|remove)\s*\(")

# Names that mean api/ is touching storage directly rather than through a
# repository method: the SQL toolkit, the tables, the engine, raw SQL text and
# the search engine's client. Asserted over the source text of every api/
# module, so a lazy import inside a function is caught as well.
API_STORAGE_MARKERS: tuple[str, ...] = (
    "sqlalchemy",
    "database.postgres.tables",
    "database.postgres.engine",
    "database.elasticsearch",
    "database.graph",
    "database.session_context",
    "database.knowledgebase_operations",
    "elasticsearch",
    " text(",
)


def classify(module: str) -> tuple[int, str] | None:
    """(layer, unit) for a first-party module, or None if it is not ours."""
    best: tuple[str, int, str] | None = None
    for prefix, layer, unit in MODULE_LAYERS:
        if (
            module == prefix or module.startswith(prefix + ".")
        ) and (best is None or len(prefix) > len(best[0])):
            best = (prefix, layer, unit)
    return (best[1], best[2]) if best else None


def is_violation(src: str, dst: str) -> str | None:
    """Why importing `dst` from `src` breaks the rule, or None if it is fine."""
    s, d = classify(src), classify(dst)
    if s is None or d is None:
        return None

    if s[1] == d[1]:            # inside its own unit
        return None

    allowed = VENDOR_SHIMS.get(src)
    if allowed is not None:
        if d[1] in allowed:
            return None
        return f"vendor shim may only reach {sorted(allowed)}, not {d[1]}"
    if d[0] > s[0]:             # strictly downward
        # HTTP may run pipelines, but it may not call any service unit. Test
        # the layer rather than enumerating today's slots so a future slot or
        # support module cannot silently create a new API bypass.
        if s[1] == "api" and d[0] == SERVICE:
            return f"forbidden: api may not reach service ({dst})"
        banned = FORBIDDEN_DOWNWARD.get(s[1])
        if banned is not None and d[1] in banned[0]:
            exempt = banned[1]
            if not any(dst == e or dst.startswith(e + ".") for e in exempt):
                return f"forbidden: {s[1]} may not reach {d[1]} ({dst})"
        return None
    if d[0] == s[0]:
        return f"sideways: {LAYER_NAMES[s[0]]} {s[1]} -> {d[1]}"
    return f"upward: {LAYER_NAMES[s[0]]} -> {LAYER_NAMES[d[0]]}"


SLOTS = frozenset(
    unit
    for _, layer, unit in MODULE_LAYERS
    if layer == SERVICE and not unit.startswith("_")
)


def slot_roots() -> dict[str, set[str]]:
    """The module paths that are a slot's public front door.

    Importing one of these is reaching for the slot; importing anything deeper
    is reaching past its contract into an implementation, which is what makes a
    slot unswappable.
    """
    roots: dict[str, set[str]] = {}
    for prefix, layer, unit in MODULE_LAYERS:
        if layer == SERVICE:
            roots.setdefault(unit, set()).add(prefix)
    return roots
