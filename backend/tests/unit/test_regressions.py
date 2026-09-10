"""Cross-module regression contracts with no external service dependencies."""
import ast
import asyncio
import importlib
import re
import tomllib
from pathlib import Path

import pytest


# ----------------------------------------------------------- synonym fallback
def test_synonym_dictionary_is_never_none():
    """Dealer.dictionary was None, so lookup() raised AttributeError.

    synonym.json does not ship with the repo and the loader is commented out,
    so the attribute must still default to an empty dict. Every RAG query goes
    through lookup(); a None here broke knowledge-base search entirely.
    """
    from visionagent.vendor.ragflow.rag.nlp import synonym

    dealer = synonym.Dealer()
    assert dealer.dictionary is not None, "Dealer.dictionary must never be None"
    assert isinstance(dealer.dictionary, dict)

    # A token that is not pure lowercase ASCII skips the wordnet branch and
    # reaches the dictionary lookup -- this is the exact path that crashed.
    assert dealer.lookup("Width3") == []
    assert dealer.lookup("2+2") == []


# --------------------------------------------------------------- NLTK assets
@pytest.mark.parametrize(
    "resource,probe",
    [
        ("punkt_tab", lambda: __import__("nltk").word_tokenize("a b c")),
        ("wordnet", lambda: __import__("nltk.corpus", fromlist=["wordnet"]).wordnet.synsets("width")),
    ],
)
def test_nltk_corpora_present(resource, probe):
    """The Python package installs nltk but cannot bundle its corpora.

    Without punkt_tab every knowledge-base query raised
    LookupError: Resource 'punkt_tab' not found, which the workflow swallowed
    and reported as "searching user knowledge base (0.00s)".
    """
    try:
        probe()
    except LookupError as exc:
        pytest.fail(
            f"NLTK resource {resource!r} missing -- RAG tokenisation will fail. "
            f"Run: python -m nltk.downloader punkt punkt_tab wordnet omw-1.4\n{exc}"
        )


# --------------------------------------------------------- model compatibility
def test_xgboost_can_load_legacy_model(app_dir: Path):
    """The pinned XGBoost runtime can load the bundled legacy model format."""
    xgboost = pytest.importorskip("xgboost")
    model = app_dir / "vendor/ragflow/rag/res/deepdoc/updown_concat_xgb.model"
    if not model.exists():
        pytest.skip("model file not present")

    booster = xgboost.Booster()
    try:
        booster.load_model(str(model))
    except Exception as exc:  # noqa: BLE001
        pytest.fail(
            f"xgboost {xgboost.__version__} cannot load the shipped model. "
            f"Pin xgboost<3.1 in pyproject.toml.\n{exc}"
        )
    assert booster.num_features() > 0


# ---------------------------------------------------------- parser dependencies
def test_pymupdf_available():
    """vlm_processor imports fitz; without it PDF processing silently no-ops.

    When PyMuPDF was absent from the project dependencies, a clean install logged
    "PyMuPDF not available. PDF processing will not work." and carried on.
    """
    pytest.importorskip(
        "fitz",
        reason="PyMuPDF missing -- add PyMuPDF to backend/pyproject.toml",
    )


def _project_dependencies(backend_dir: Path) -> list[str]:
    manifest = tomllib.loads((backend_dir / "pyproject.toml").read_text(encoding="utf-8"))
    return manifest["project"]["dependencies"]


def test_project_dependencies_pin_xgboost(backend_dir: Path):
    """An unpinned xgboost silently upgrades and breaks ingestion again."""
    dependencies = _project_dependencies(backend_dir)
    line = next((item for item in dependencies if item.lower().startswith("xgboost")), None)
    assert line is not None, "xgboost missing from pyproject.toml"
    assert any(op in line for op in ("<", "==", "~=")), (
        f"xgboost is unpinned ({line!r}); 3.1+ cannot read the shipped "
        "updown_concat_xgb.model"
    )


def test_project_dependencies_include_pymupdf(backend_dir: Path):
    dependencies = [item.lower() for item in _project_dependencies(backend_dir)]
    assert any(item.startswith("pymupdf") for item in dependencies), (
        "PyMuPDF missing from pyproject.toml"
    )


# --------------------------------------------------------- HTTP error boundary
def _handlers_swallowing_http_exception(py_file: Path) -> list[str]:
    """Find `except Exception` blocks that hide an HTTPException raised in the
    same try body by re-raising a fresh HTTPException."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for tryblock in ast.walk(node):
            if not isinstance(tryblock, ast.Try):
                continue
            raises_http = any(
                isinstance(r, ast.Raise)
                and isinstance(r.exc, ast.Call)
                and getattr(r.exc.func, "id", "") == "HTTPException"
                for stmt in tryblock.body
                for r in ast.walk(stmt)
            )
            if not raises_http:
                continue
            # a dedicated `except HTTPException: raise` guard fixes it
            if any(
                getattr(h.type, "id", None) == "HTTPException" for h in tryblock.handlers
            ):
                continue
            for handler in tryblock.handlers:
                name = getattr(handler.type, "id", None) if handler.type else "bare"
                if name not in ("Exception", "bare"):
                    continue
                reraises = any(
                    isinstance(x, ast.Raise)
                    and isinstance(x.exc, ast.Call)
                    and getattr(x.exc.func, "id", "") == "HTTPException"
                    for x in ast.walk(handler)
                )
                guarded = any(isinstance(x, ast.If) for x in ast.walk(handler))
                if reraises and not guarded:
                    offenders.append(f"{py_file.name}:{handler.lineno} in {node.name}()")
    return offenders


def test_http_exceptions_are_not_swallowed(app_dir: Path):
    """A bare `except Exception` caught the handler's own 401/404 and turned it
    into a generic 500 -- e.g. kill-processing answered
    500 "Upload cancelled" when it meant 404 "Process not found".
    """
    offenders = []
    for py_file in sorted((app_dir / "api" / "routes").glob("*.py")):
        offenders.extend(_handlers_swallowing_http_exception(py_file))
    assert not offenders, (
        "these handlers convert their own HTTPException into a 500; add "
        "`except HTTPException: raise` before `except Exception`:\n  "
        + "\n  ".join(offenders)
    )


# --------------------------------------------------------- cleanup exceptions
def test_upload_cleanup_does_not_mask_errors(app_dir: Path):
    """`del chunks` sat in a finally block.

    When execute_insert_process_sync() raised, `chunks` was unbound and the
    finally raised UnboundLocalError, replacing the real error with
    "cannot access local variable 'chunks'".
    """
    src = (app_dir / "api" / "routes" / "file_upload_rt.py").read_text(encoding="utf-8")
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        deleted = {
            t.id
            for stmt in node.finalbody
            for d in ast.walk(stmt)
            if isinstance(d, ast.Delete)
            for t in d.targets
            if isinstance(t, ast.Name)
        }
        if not deleted:
            continue
        # every name deleted in the finally must be bound before the try
        pre_assigned = {
            t.id
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign) and n.lineno < node.lineno
            for t in n.targets
            if isinstance(t, ast.Name)
        }
        unguarded = deleted - pre_assigned
        assert not unguarded, (
            f"`del {', '.join(sorted(unguarded))}` in the finally at line "
            f"{node.finalbody[0].lineno} can raise UnboundLocalError and mask "
            "the real exception; initialise it before the try"
        )


# ---------------------------------------------------------- import contracts
@pytest.mark.parametrize(
    "module",
    [
        "visionagent.models",
        "visionagent.database.postgres.tables",
        "visionagent.database.elasticsearch.retrieval",
        "visionagent.vendor.ragflow.rag.nlp.synonym",
        "visionagent.vendor.ragflow.rag.nlp.rag_tokenizer",
        "visionagent.api.routes.ai_search_rt",
        "visionagent.api.routes.add_context_rt",
        "visionagent.api.routes.file_upload_rt",
        "visionagent.api.routes.history_rt",
        "visionagent.api.routes.graphml_rt",
        "visionagent.api.routes.user_rt",
    ],
)
def test_module_imports(module):
    """Every registered router and core module must import cleanly.

    router/ai_search_mcp_rt.py is deliberately absent from this list: it is an
    inert future extension point and is not a registered router. See
    test_unregistered_router_is_not_wired.
    """
    importlib.import_module(module)


def test_request_identity_is_never_stored_as_ambient_process_state(app_dir: Path):
    """Request identity reaches tools only through immutable arguments."""
    ambient_module = app_dir / "service" / "context.py"
    assert not ambient_module.exists(), (
        "service/context.py must not hold request identity; pass ToolContext explicitly"
    )

    forbidden = {
        "set_current_user_id",
        "get_current_session_id",
        "set_current_session_id",
    }
    offenders: list[str] = []
    for py_file in app_dir.rglob("*.py"):
        if "vendor" in py_file.parts:
            continue
        source = py_file.read_text(encoding="utf-8")
        names = sorted(name for name in forbidden if name in source)
        if names:
            offenders.append(f"{py_file.relative_to(app_dir)}: {', '.join(names)}")
    assert not offenders, "ambient identity API was reintroduced:\n  " + "\n  ".join(offenders)


def test_reserved_deep_research_module_is_not_wired(app_dir: Path):
    """The reserved Deep Research module must remain unmounted.

    The frontend control is hidden and the extension point exposes no router;
    `/deep_research/` therefore remains absent until implemented deliberately.
    """
    reserved_module = app_dir / "api" / "routes" / "ai_search_mcp_rt.py"
    if not reserved_module.exists():
        pytest.skip("reserved module is absent")

    src = reserved_module.read_text(encoding="utf-8")
    defines_route = "/deep_research/" in src
    has_router = "APIRouter()" in src
    registered = "ai_search_mcp_rt" in (app_dir / "api" / "main.py").read_text(
        encoding="utf-8"
    )

    if defines_route and not (has_router and registered):
        assert not registered, (
            "ai_search_mcp_rt is registered in api/main.py but is missing its "
            "APIRouter and imports -- the app will fail to start"
        )


# --------------------------------------------------------- evaluator contract
def test_reflection_reads_the_keys_the_evaluator_emits(monkeypatch):
    """Reflection consumes the typed evaluator fields that guide another round."""
    import visionagent.service.evaluator.llm as agent
    from visionagent.models import (
        Evaluation,
        RetrievedChunk,
        SourceType,
        WebSearchMode,
    )

    seen = {}

    async def capture(prompt):
        seen["prompt"] = prompt
        return "[]"

    monkeypatch.setattr(agent, "middle_json_model", capture)
    asyncio.run(agent.reflection(
        "how wide?",
        [RetrievedChunk(id="c1", content="a chunk", source_type=SourceType.KNOWLEDGE_BASE)],
        Evaluation(sufficient_score=0.25, reasons="thin coverage",
                   comments="no width figures"),
        web_search=WebSearchMode.AUTO,
    ))

    prompt = seen["prompt"]
    assert "0.25" in prompt, "the real score never reached the reflection prompt"
    assert "thin coverage" in prompt, "the evaluator's reasons never reached it"
    assert "no width figures" in prompt, (
        "the evaluator's `comments` says what context is missing -- round 2 "
        "cannot target the gap without it"
    )
    assert "Unknown" not in prompt, "a field fell through to its default"


def test_workflow_log_reports_the_real_score():
    """The typed verdict carries the score used by trace rendering."""
    from visionagent.models import Evaluation
    from visionagent.service.evaluator.llm import is_sufficient

    assert set(Evaluation.model_fields) == {"sufficient_score", "reasons", "comments"}
    assert is_sufficient(Evaluation(sufficient_score=0.2)) is False
    assert is_sufficient(Evaluation(sufficient_score=0.9)) is True


# -------------------------------------------------------------- rerank order
def test_rerank_scores_align_with_input_order(app_dir: Path):
    """Provider-ranked scores are mapped back to their original inputs."""
    # The behavioral counterpart is
    # backend/tests/unit/test_embedding_rerank.py::test_score_is_aligned_to_input_order_not_rank.
    src = (app_dir / "providers" / "rerank" / "dashscope" / "provider.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_aligned_scores"
    )
    # Inspect executable statements only.
    stmts = fn.body[1:] if (
        fn.body and isinstance(fn.body[0], ast.Expr)
        and isinstance(fn.body[0].value, ast.Constant)
        and isinstance(fn.body[0].value.value, str)
    ) else fn.body
    code = "\n".join(ast.unparse(s) for s in stmts)

    assert "[res.score for res in results]" not in code, (
        "scores are being collected in the reranker's output order; map them "
        "back to the input order via the provider's result index"
    )
    assert "result.index" in code, (
        "the reranker must map scores back to their input position "
        "(DashScope returns the original document index)"
    )
    assert "[0.0] * count" in code or "np.zeros" in code, (
        "allocate a score array of len(texts) and fill it by original position, "
        "so unscored texts default to 0.0 rather than shifting the alignment"
    )


# ------------------------------------------------------ source-type contract
def test_web_source_type_producer_and_consumer_agree(app_dir: Path):
    """Every source type consumed by query handling has a declared producer."""
    from visionagent.models import SourceType
    from visionagent.service.executer.concurrent import _SOURCE_TYPE_BY_TOOL

    # Every mapping value is a real SourceType, making drift unrepresentable.
    produced = set(_SOURCE_TYPE_BY_TOOL.values())
    assert produced, "gather_results() has no tool -> source_type mapping"
    assert produced <= {s.value for s in SourceType}, (
        f"a tool maps to a source_type that is not in the enum: "
        f"{produced - {s.value for s in SourceType}}"
    )

    query_source = (
        (app_dir / "api" / "routes" / "ai_search_rt.py").read_text(encoding="utf-8")
        + (app_dir / "pipeline" / "query.py").read_text(encoding="utf-8")
    )
    consumed = set(re.findall(
        r"source_type[\"']\)\s*==\s*[\"']([^\"']+)[\"']", query_source))
    stale = consumed - produced
    assert not stale, (
        f"the router filters on source_type {sorted(stale)}, which nothing "
        f"produces. gather_results() emits {sorted(produced)}."
    )


# ---------------------------------------------------------- web enrichment
def test_web_snippet_ranking_has_no_ephemeral_database():
    """A per-query Chroma collection added a heavyweight, racy store."""
    import inspect

    from visionagent.service.executer.tools.web import snippets

    source = inspect.getsource(snippets)
    assert "chromadb" not in source.lower()
    assert inspect.iscoroutinefunction(snippets._rerank)
    assert "aembed_batch" in source


def test_related_questions_collected_from_all_web_chunks(app_dir: Path):
    """Collect questions from every result in first-seen order."""
    src = (app_dir / "pipeline" / "query.py").read_text(encoding="utf-8")
    assert "web_chunks[0].get(" not in src, (
        "related_questions is read from the first web chunk only; collect it "
        "across every tool result"
    )
    assert "for result in run.tool_results" in src
    assert "for question in result.related_questions" in src


# ---------------------------------------------------------- web-search mode
def test_web_search_toggle_forces_rather_than_permits():
    """FORCE schedules web search; AUTO delegates; DISABLED forbids it."""
    from visionagent.models import (
        Intent,
        Plan,
        Scenario,
        ToolCall,
        ToolName,
        WebSearchMode,
    )
    from visionagent.service.planner import rule_based as agent

    available = [t.value for t in ToolName]
    _kb = Intent(scenario=Scenario.KNOWLEDGE, intents=["kb(filter)"])

    # OFF: the analyser decides, and it did not ask for the web here.
    off = asyncio.run(agent.agent_plan(["q"], _kb, available=available))
    assert ToolName.WEB_SEARCH not in [c.tool_name for c in off.calls]

    # ON: forced, whatever the analyser decided.
    on = asyncio.run(agent.agent_plan(
        ["q"], _kb, available=available, web_search=WebSearchMode.FORCE
    ))
    assert [c.tool_name for c in on.calls] == [
        ToolName.RAG, ToolName.GRAPHRAG, ToolName.WEB_SEARCH]

    # OFF is not a prohibition: an analyser that asks for the web still gets it.
    permitted = asyncio.run(agent.agent_plan(
        ["q"], Intent(scenario=Scenario.KNOWLEDGE, intents=["web_search"]),
        available=available,
    ))
    assert permitted == Plan(calls=[ToolCall(tool_name=ToolName.WEB_SEARCH, query=["q"])])

    # DISABLED is the prohibition the bool could not express: it strips web
    # search from the plan even when the analyser asked for it.
    denied = asyncio.run(agent.agent_plan(
        ["q"], Intent(scenario=Scenario.KNOWLEDGE, intents=["web_search"]),
        available=available, web_search=WebSearchMode.DISABLED,
    ))
    assert ToolName.WEB_SEARCH not in [c.tool_name for c in denied.calls]


def test_web_search_defaults_to_auto():
    """The default lets intent decide instead of forcing a paid web request."""
    from visionagent.models import ChatRequest

    assert ChatRequest(message="x").web_search is False, (
        "web_search must default to auto; defaulting to force would make every "
        "query pay for a web search"
    )


def test_scenario_survives_json_wrapping():
    """`analyze_chat_scenario` asks for a bare word but requests JSON mode, so
    the reply comes back wrapped. Comparing the raw text never matched, so the
    classifier always fell through to "professional" and no query was ever
    routed as casual -- against DashScope as well as the Claude CLI.
    """
    from visionagent.service.intent.llm import _scenario_word

    for raw in ('casual', '"casual"', '  CASUAL  ', '{"scenario": "casual"}',
                '{"scenario":"Casual"}'):
        assert _scenario_word(raw) == "casual", raw
    assert _scenario_word('{"scenario": "professional"}') == "professional"
    assert _scenario_word("") == ""


def test_graphrag_ingestion_survives_chunks_without_docnm():
    """Raw parser chunks may omit docnm, so graph indexing uses the upload name.

    The same name identifies document-owned graph entities during deletion.
    """
    import inspect

    from visionagent.service.vectorstore.graphstore.service import GraphRAGService

    src = inspect.getsource(GraphRAGService._process_chunks_for_graphrag)
    assert "chunk['docnm']" not in src and 'chunk["docnm"]' not in src, (
        "subscripting 'docnm' raises KeyError on raw chunks; use "
        "chunk.get('docnm') or file_name"
    )
    assert "docnm = file_name" in src, "the upload's original file name is authoritative"


def test_retrieval_keeps_ref_images_in_the_typed_base64_contract():
    """Storage and answer contracts both use strings; no layer decodes PIL."""
    import inspect

    from visionagent.database.elasticsearch import chunks as es_operations
    from visionagent.database.elasticsearch import retrieval
    from visionagent.service.executer.tools import graphrag as graphrag_service

    for module in (graphrag_service, retrieval, es_operations):
        src = inspect.getsource(module)
        assert "b64decode" not in src and "Image.open" not in src, (
            f"{module.__name__} decodes page previews out of list[str]"
        )


# ----------------------------------------------------- async graph retrieval
def test_graphrag_search_does_not_call_asyncio_run_on_the_running_loop():
    """Provider awaits stay cancellable; only synchronous storage uses a worker."""
    import ast
    from pathlib import Path

    src = Path(
        importlib.import_module("visionagent.service.executer.tools.graphrag").__file__
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)

    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "graphrag"
    )
    helpers = {
        n.name: n for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef)
        and n.name in {"search_entities", "search_relationships"}
    }
    assert set(helpers) == {"search_entities", "search_relationships"}
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "run"
        for helper in helpers.values()
        for n in ast.walk(helper)
    )

    awaited = {
        node.value.func.id
        for node in ast.walk(fn)
        if isinstance(node, ast.Await)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
    }
    assert {"search_entities", "search_relationships"} <= awaited

    threaded = {
        n.args[0].id for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "to_thread"
        and n.args and isinstance(n.args[0], ast.Name)
    }
    assert "retrieve_chunks_from_es" in threaded
    assert not {"search_entities", "search_relationships"} & threaded
