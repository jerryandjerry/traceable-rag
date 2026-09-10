"""Contract tests for visionagent.models.

Every internal contract sets `extra="forbid"` and bounds its numeric fields, so
a wrong shape fails at the component boundary with a ValidationError naming the
field rather than three frames later with a KeyError. These tests assert that
the guard rails are actually on -- a model that silently accepts anything is
worse than no model, because it looks like a contract.
"""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

import visionagent.models as models
from visionagent.models import (
    AgentState,
    Answer,
    Entity,
    Evaluation,
    Evidence,
    IngestRun,
    Intent,
    ParsedChunk,
    Plan,
    Relation,
    RetrievedChunk,
    Scenario,
    SourceType,
    StepStatus,
    ToolCall,
    ToolContext,
    ToolName,
    ToolResult,
    TraceStep,
)


def _job(**kw):
    """A minimal authorized ticket for state tests."""
    from visionagent.models import AuthorizationScope, JobIdentity, QueryJob

    return QueryJob(
        identity=JobIdentity(run_id=kw.get("run_id", "r1"), user_id="u1"),
        authorization=AuthorizationScope(),
        session_id="s1",
        question="q",
    )

# Contracts internal to the pipeline. The api.* models are excluded: they are
# HTTP bodies, and rejecting unknown keys there would break clients that send
# extra fields.
INTERNAL = [
    ParsedChunk, RetrievedChunk, Evaluation, Intent, Plan, ToolCall, ToolContext, ToolResult,
    Answer, Entity, Relation, TraceStep, Evidence, AgentState, IngestRun,
]


@pytest.mark.parametrize("model", INTERNAL, ids=lambda m: m.__name__)
def test_internal_contracts_forbid_unknown_fields(model: type[BaseModel]):
    assert model.model_config.get("extra") == "forbid", (
        f"{model.__name__} accepts unknown keys, so a typo or a stale producer "
        "passes validation and fails somewhere downstream instead"
    )


def test_every_exported_name_resolves():
    """__all__ and the actual exports must not drift."""
    missing = [n for n in models.__all__ if not hasattr(models, n)]
    assert not missing, f"__all__ names nothing: {missing}"


# ------------------------------------------------------------------- chunks
def test_parsed_chunk_round_trips():
    c = ParsedChunk(id="c1", content="Sidewalk widths.", content_tokens="sidewalk width",
                    page_nums=[3], top_offsets=[120])
    assert ParsedChunk(**c.model_dump()) == c
    assert c.page_num == 3 and c.top_offset == 120


def test_parsed_chunk_rejects_the_raw_deepdoc_keys():
    """The adapter must map DeepDoc fields to the typed chunk schema."""
    with pytest.raises(ValidationError) as exc:
        ParsedChunk(id="c1", content="x", page_num_int=[1], top_int=[10])
    assert "page_num_int" in str(exc.value)


def test_parsed_chunk_requires_non_empty_content():
    with pytest.raises(ValidationError):
        ParsedChunk(id="c1", content="")


def test_parsed_chunk_page_num_is_derived_not_settable():
    """page_num/top_offset are the first element of their list, so a caller
    cannot set one inconsistently with the other."""
    with pytest.raises(ValidationError):
        ParsedChunk(id="c1", content="x", page_num=1)


@pytest.mark.parametrize("score", [-0.01, 1.01, 2.0])
def test_retrieved_chunk_score_is_bounded(score: float):
    """Scores are a similarity in [0, 1]; anything else is a unit mix-up."""
    with pytest.raises(ValidationError):
        RetrievedChunk(id="c1", content="x", source_type=SourceType.KNOWLEDGE_BASE,
                       score=score)


def test_retrieved_chunk_carries_its_own_score():
    """Regression: the reranker returned scores in reranked order and they were
    zipped against the original order, so a score could land on the wrong chunk.
    Carrying the score on the chunk makes that unrepresentable."""
    chunks = [
        RetrievedChunk(id=f"c{i}", content=f"body {i}",
                       source_type=SourceType.KNOWLEDGE_BASE, score=i / 10)
        for i in range(3)
    ]
    reordered = sorted(chunks, key=lambda c: -c.score)
    assert [(c.id, c.score) for c in reordered] == [("c2", 0.2), ("c1", 0.1), ("c0", 0.0)]


def test_source_type_values_match_what_the_frontend_switches_on():
    assert {s.value for s in SourceType} == {
        "knowledge_base", "web_search", "current_context"
    }


# -------------------------------------------------------------------- query
def test_tool_context_is_immutable_and_rejects_unknown_identity_fields():
    context = ToolContext(run_id="r1", user_id="u1", session_id="s1")
    with pytest.raises(ValidationError):
        context.user_id = "attacker"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ToolContext(run_id="r1", user_id="u1", session_id="s1", tenant="other")


@pytest.mark.parametrize("field", ["run_id", "user_id", "session_id"])
def test_tool_context_requires_every_scope_identifier(field: str):
    values = {"run_id": "r1", "user_id": "u1", "session_id": "s1"}
    values[field] = ""
    with pytest.raises(ValidationError):
        ToolContext(**values)


def test_tool_name_values_match_the_dispatch_branches():
    """These strings appear in the planner prompt and the if/elif chain."""
    assert {t.value for t in ToolName} == {"RAG", "GraphRAG", "web_search", "LLM"}


def test_plan_uses_reports_scheduled_tools():
    plan = Plan(calls=[ToolCall(tool_name=ToolName.RAG, query=["curb width"])])
    assert plan.uses(ToolName.RAG)
    assert not plan.uses(ToolName.WEB_SEARCH)


def test_tool_call_requires_a_query():
    with pytest.raises(ValidationError):
        ToolCall(tool_name=ToolName.RAG, query=[])


def test_unknown_tool_name_is_rejected():
    with pytest.raises(ValidationError):
        ToolCall(tool_name="deep_research", query=["x"])


def test_evaluation_field_names_are_the_ones_the_evaluator_emits():
    """Regression: the reflection prompt read score/sufficient/missing while the
    evaluator emitted sufficient_score/reasons/comments, so every reflection ran
    on 0.0 and "Unknown"."""
    assert set(Evaluation.model_fields) == {"sufficient_score", "reasons", "comments"}


def test_evaluation_is_data_not_a_configuration_policy():
    """The evaluator, not a model import, owns the routing threshold."""
    evaluation = Evaluation(sufficient_score=0.5, reasons="thin")
    assert evaluation.sufficient_score == 0.5
    assert not hasattr(evaluation, "is_sufficient")


def test_evaluation_score_is_bounded():
    with pytest.raises(ValidationError):
        Evaluation(sufficient_score=1.5)


def test_tool_result_can_carry_a_failure_instead_of_raising():
    """Tools are gathered concurrently; one failing must not end the round."""
    r = ToolResult(tool_name=ToolName.WEB_SEARCH, error="serper timed out", timed_out=True)
    assert r.chunks == [] and r.error


def test_intent_defaults_are_empty_not_none():
    i = Intent(scenario=Scenario.KNOWLEDGE)
    assert i.intents == [] and i.keywords_high == [] and i.keywords_low == []


# -------------------------------------------------------------------- trace
def test_trace_step_is_flat_with_a_parent_pointer():
    """Nested children cannot be patched; SSE sends deltas over a ~50s run."""
    assert "parent_id" in TraceStep.model_fields
    assert "children" not in TraceStep.model_fields


def test_trace_duration_cannot_be_negative():
    """time.perf_counter() is monotonic precisely so this cannot happen; the
    bound catches a caller that swaps in time.time()."""
    with pytest.raises(ValidationError):
        TraceStep(id="s1", label="searching", duration_s=-0.5)


def test_trace_step_starts_pending():
    assert TraceStep(id="s1", label="initial search").status is StepStatus.PENDING


def test_evidence_can_carry_a_thumbnail():
    """The whole reason the trace stops being a string."""
    e = Evidence(step_id="s1", kind="chunk", label="Curb Design Guide.pdf",
                 chunk_id="c1", thumbnail="data:image/webp;base64,AAA")
    assert e.thumbnail


# ---------------------------------------------------------------------- run
def test_agent_state_round_is_a_counter_bounded_by_settings_not_by_the_model():
    """The ceiling is MAX_ROUNDS (default 2, counting the initial search) and
    the pipeline's loop enforces it. The model only refuses a negative count --
    a bound baked in here would fight the setting the moment it is raised."""
    from visionagent.config.settings import settings

    assert settings.max_rounds == 2, "default is one initial search plus one reflection"

    for r in (0, 1, 5):
        AgentState(job=_job(), round=r)
    with pytest.raises(ValidationError):
        AgentState(job=_job(), round=-1)


def test_agent_state_has_one_immutable_identity_source():
    """Identity comes from the ticket the API issued, and only from there.

    Not stored on the state and not settable: run.context is derived, so it
    cannot drift from the job, and the job binding is frozen so a turn cannot
    be handed a different identity halfway through.
    """
    j = _job()
    run = AgentState(job=j)

    assert run.job is j
    assert run.context == ToolContext(run_id="r1", user_id="u1", session_id="s1")
    assert (run.run_id, run.user_id, run.session_id) == ("r1", "u1", "s1")
    assert not {"run_id", "user_id", "session_id"} & set(AgentState.model_fields)
    with pytest.raises(ValidationError):
        run.job = _job(run_id="r2")


def test_query_run_is_not_passed_to_components():
    """AgentState is orchestration state. If a component ever accepted it, that
    component could no longer be called standalone -- which is the property the
    Protocol layout exists to provide. Pinned as a design decision."""
    import inspect

    from visionagent.models import run as run_module

    for obj in vars(run_module).values():
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
            for field in obj.model_fields.values():
                assert field.annotation is not AgentState or obj is AgentState


def test_ingest_run_counts_cannot_be_negative():
    with pytest.raises(ValidationError):
        IngestRun(run_id="r1", user_id="u1", file_name="f.pdf", indexed_count=-1)


# -------------------------------------------------------------------- graph
def test_entity_and_relation_require_names():
    with pytest.raises(ValidationError):
        Entity(name="")
    with pytest.raises(ValidationError):
        Relation(source="a", target="")
