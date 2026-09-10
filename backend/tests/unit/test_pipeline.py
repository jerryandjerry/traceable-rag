"""Tests for pipeline/ -- the tracer and the query orchestration.

Intended function of the tracer:
    step(label)  -> a timed, nested TraceStep whose duration is MEASURED, never
                    derived from its children
    emit(...)    -> an Evidence row attached to the currently open step
    durations use perf_counter: monotonic, wall time, never negative

Intended function of the query pipeline:
    dispatch(plan) -> exactly one ToolResult per call, gathered CONCURRENTLY,
                      a failure or timeout confined to its own step
    merge(results) -> exactly the union, de-duplicated by id, first-seen order
    run(query)     -> intent, plan, ranked context and evaluation, with a
                      second round only when the evaluator asks for one
"""
from __future__ import annotations

import asyncio
import time

import pytest

from visionagent.models import (
    AgentState,
    AuthorizationScope,
    Evaluation,
    Intent,
    JobIdentity,
    Plan,
    QueryJob,
    RetrievedChunk,
    Scenario,
    SourceType,
    StepStatus,
    ToolCall,
    ToolContext,
    ToolName,
    ToolResult,
    TurnOptions,
    WebSearchMode,
)
from visionagent.pipeline.query import QueryPipeline
from visionagent.pipeline.trace import Tracer
from visionagent.service.executer import ConcurrentExecuter, gather_results
from visionagent.service.executer.tools import ToolRegistry


def chunk(cid: str, body: str = "body") -> RetrievedChunk:
    return RetrievedChunk(id=cid, content=body, source_type=SourceType.KNOWLEDGE_BASE)


class StubTool:
    """A tool is a callable: the registry holds the functions themselves."""

    def __init__(self, name: ToolName, chunks=None, delay=0.0, error=None, hang=False):
        self._name, self._chunks = name, chunks or []
        self._delay, self._error, self._hang = delay, error, hang
        self.contexts: list[ToolContext] = []

    async def __call__(self, query: str, *, context: ToolContext, emit=None) -> ToolResult:
        self.contexts.append(context)
        if self._hang:
            await asyncio.sleep(30)
        await asyncio.sleep(self._delay)
        if self._error:
            return ToolResult(tool_name=self._name, error=self._error)
        return ToolResult(tool_name=self._name, chunks=list(self._chunks))


class StubIntent:
    """Mirrors the IntentParser Protocol, which names analyze_query_intent."""

    def __init__(self, intents, scenario="professional"):
        self._intents = intents
        self._scenario = scenario

    async def analyze_query_intent(self, query):
        return Intent(scenario=Scenario.KNOWLEDGE, intents=self._intents)

    async def analyze_chat_scenario(self, question):
        return self._scenario


class StubAnswer:
    """Mirrors the AnswerGenerator Protocol. Frames are opaque strings to the
    pipeline; "event: end" is the marker that closes the answer step."""

    def __init__(self):
        self.calls: list[str] = []
        self.professional_contexts: list[list[RetrievedChunk]] = []
        self.casual_contexts: list[list[RetrievedChunk]] = []
        self.citation_ids: list[dict[str, str]] = []

    async def get_chat_completion(self, session_id, question, context_list, user_id,
                                  final_prompt, related_questions, snippets, *,
                                  run_id, citation_ids, media=None):
        self.calls.append("professional")
        self.professional_contexts.append(context_list)
        self.citation_ids.append(dict(citation_ids))
        yield "data: token\n\n"
        yield "event: end\ndata: {}\n\n"

    async def casual_chat_completion(self, session_id, question, user_id,
                                     final_prompt, web_context=None, *,
                                     run_id, related_questions=None):
        self.calls.append("casual")
        self.casual_contexts.append(web_context or [])
        yield "data: hi\n\n"
        yield "event: end\ndata: {}\n\n"


class StubReranker:
    """Injected reranker slot that keeps pipeline tests provider-free."""

    name = "stub"

    async def score(self, *, query, texts): return [1.0] * len(texts)


@pytest.fixture(autouse=True)
def _no_live_reranker(monkeypatch):
    import visionagent.service.rerank as agent

    monkeypatch.setattr(agent, "build_reranker", lambda *a, **k: StubReranker())


class StubEvaluator:
    """Implements the Evaluator protocol used by QueryPipeline."""

    def __init__(self, score=1.0, follow_up=None):
        self._score, self._follow_up = score, follow_up

    async def evaluate_context_sufficiency(self, context_list, query):
        return Evaluation(sufficient_score=self._score)

    def is_sufficient(self, evaluation):
        return evaluation.sufficient_score > 0.5

    async def reflection(self, user_query, context_list, evaluation,
                         web_search=WebSearchMode.FORCE, available=None):
        return self._follow_up


@pytest.fixture(autouse=True)
def _no_db_history(monkeypatch):
    """stream()'s answer stage reads the session history for the prompt; unit
    tests have no database."""
    import visionagent.pipeline.query as q

    monkeypatch.setattr(q, "get_user_history_questions", lambda sid: [])


def pipeline(tools: dict, **kw) -> QueryPipeline:
    kw.setdefault("intent", StubIntent(["kb(filter)"]))
    kw.setdefault("evaluator", StubEvaluator())
    kw.setdefault("answer", StubAnswer())
    kw.setdefault("reranker", StubReranker())
    # The timeout bounds one tool call, so it belongs to the executer that
    # makes the call rather than to the pipeline that asks for the round.
    timeout = kw.pop("tool_timeout_s", None)
    return QueryPipeline(
        executer=ConcurrentExecuter(registry=ToolRegistry(tools),
                                    tool_timeout_s=timeout),
        **kw,
    )


def context(run_id="r", user_id="u", session_id="s") -> ToolContext:
    return ToolContext(run_id=run_id, user_id=user_id, session_id=session_id)


def job(question="q", *, run_id="r", user_id="u", session_id="s",
        web_search=WebSearchMode.AUTO, allowed=None) -> QueryJob:
    return QueryJob(
        identity=JobIdentity(run_id=run_id, user_id=user_id),
        authorization=AuthorizationScope(
            allowed_tools=frozenset(ToolName) if allowed is None else frozenset(allowed)
        ),
        session_id=session_id,
        question=question,
        effective=TurnOptions(web_search=web_search),
    )


def state(question="q", *, run_id="r", user_id="u", session_id="s",
          web_search=WebSearchMode.AUTO, allowed=None, **kwargs) -> AgentState:
    return AgentState(
        job=job(question, run_id=run_id, user_id=user_id, session_id=session_id,
                web_search=web_search, allowed=allowed),
        **kwargs,
    )


# ==================================================================== tracer
def test_step_durations_are_measured_and_never_negative():
    t = Tracer()
    with t.step("work"):
        time.sleep(0.02)
    step = t.tree()[0]
    assert step.duration_s >= 0.02
    assert step.status is StepStatus.DONE


def test_a_parents_duration_is_measured_not_summed_from_children():
    """Two 20ms children run in sequence here, so the parent is at least their
    sum -- but the number comes from the parent's own clock, which is what lets
    the gap expose queueing and contention."""
    t = Tracer()
    with t.step("parent"):
        with t.step("a"):
            time.sleep(0.02)
        with t.step("b"):
            time.sleep(0.02)
    by_label = {s.label: s for s in t.tree()}
    parent, a, b = (by_label[k].duration_s for k in ("parent", "a", "b"))
    # Sequential children, so the parent spans both. Compared with a tolerance
    # rather than exactly: durations are rounded to 4dp, and parent vs a+b sits
    # right on that boundary, which made this flake under load.
    assert parent >= (a + b) * 0.95
    assert parent > max(a, b) * 1.5, "a parent derived from max() would fail here"


def test_steps_nest_by_parent_id_not_by_containment():
    t = Tracer()
    with t.step("parent") as p:
        with t.step("child") as c:
            pass
    assert c.parent_id == p.id and p.parent_id is None


def test_a_raising_step_is_marked_failed_and_still_timed():
    t = Tracer()
    with pytest.raises(RuntimeError), t.step("boom"):
        raise RuntimeError("x")
    step = t.tree()[0]
    assert step.status is StepStatus.FAILED and step.duration_s is not None


def test_evidence_attaches_to_the_open_step():
    t = Tracer()
    with t.step("searching") as s:
        t.emit("doc", "curb.pdf")
    assert [(e.step_id, e.label) for e in t.evidence] == [(s.id, "curb.pdf")]


def test_evidence_outside_a_step_is_an_error_not_a_silent_orphan():
    with pytest.raises(RuntimeError, match="no open step"):
        Tracer().emit("doc", "curb.pdf")


def test_started_at_is_an_offset_from_the_run_not_a_wall_clock():
    t = Tracer()
    with t.step("first"):
        time.sleep(0.02)
    with t.step("second"):
        pass
    first, second = t.tree()
    assert first.started_at == pytest.approx(0.0, abs=0.01)
    assert second.started_at >= 0.02


# ================================================================== dispatch
def test_tools_run_concurrently_not_one_after_another():
    """Three 0.2-second calls complete in roughly one concurrent interval."""
    plan = Plan(calls=[
        ToolCall(tool_name=ToolName.RAG, query=["q"]),
        ToolCall(tool_name=ToolName.GRAPHRAG, query=["q"]),
        ToolCall(tool_name=ToolName.WEB_SEARCH, query=["q"]),
    ])
    p = pipeline({
        ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")], delay=0.2),
        ToolName.GRAPHRAG: StubTool(ToolName.GRAPHRAG, [chunk("b")], delay=0.2),
        ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, [chunk("c")], delay=0.2),
    })
    tracer = Tracer()
    start = time.perf_counter()
    results = asyncio.run(p.executer.run(plan, context=context(), allowed_tools=frozenset(ToolName), observer=tracer))
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    # < 0.4s, not < 0.6s: the loose bound passes even when fully serialised.
    assert elapsed < 0.4, f"tools were serialised: {elapsed:.2f}s for 3x0.2s"


def test_a_parent_step_measures_close_to_the_slowest_child_when_concurrent():
    """The honest tree: parent ~= max(children), not sum. If that gap grows,
    concurrency has stopped working."""
    plan = Plan(calls=[
        ToolCall(tool_name=ToolName.RAG, query=["q"]),
        ToolCall(tool_name=ToolName.WEB_SEARCH, query=["q"]),
    ])
    p = pipeline({
        ToolName.RAG: StubTool(ToolName.RAG, delay=0.05),
        ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, delay=0.2),
    })
    tracer = Tracer()
    with tracer.step("initial search"):
        asyncio.run(p.executer.run(plan, context=context(), allowed_tools=frozenset(ToolName), observer=tracer))

    by_label = {s.label: s for s in tracer.tree()}
    parent = by_label["initial search"].duration_s
    slowest = max(by_label["searching online"].duration_s,
                  by_label["searching user knowledge base"].duration_s)
    assert parent < slowest + 0.1, f"parent {parent:.2f}s vs slowest child {slowest:.2f}s"


def test_one_failing_tool_does_not_end_the_round():
    plan = Plan(calls=[
        ToolCall(tool_name=ToolName.RAG, query=["q"]),
        ToolCall(tool_name=ToolName.WEB_SEARCH, query=["q"]),
    ])
    p = pipeline({
        ToolName.RAG: StubTool(ToolName.RAG, [chunk("kept")]),
        ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, error="serper down"),
    })
    tracer = Tracer()
    results = asyncio.run(p.executer.run(plan, context=context(), allowed_tools=frozenset(ToolName), observer=tracer))

    assert [c.id for c in gather_results(results)] == ["kept"]
    failed = [s for s in tracer.tree() if s.status is StepStatus.FAILED]
    assert [s.label for s in failed] == ["searching online"]


def test_a_hanging_tool_times_out_instead_of_holding_the_round():
    """gather waits for its slowest member, so without a per-tool timeout one
    hung provider holds the request for the socket timeout with no signal."""
    plan = Plan(calls=[
        ToolCall(tool_name=ToolName.RAG, query=["q"]),
        ToolCall(tool_name=ToolName.WEB_SEARCH, query=["q"]),
    ])
    p = pipeline({
        ToolName.RAG: StubTool(ToolName.RAG, [chunk("kept")]),
        ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, hang=True),
    }, tool_timeout_s=0.1)

    tracer = Tracer()
    start = time.perf_counter()
    results = asyncio.run(p.executer.run(plan, context=context(), allowed_tools=frozenset(ToolName), observer=tracer))
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"the hung tool held the round for {elapsed:.1f}s"
    assert [c.id for c in gather_results(results)] == ["kept"]
    timed_out = [r for r in results if r.timed_out]
    assert len(timed_out) == 1 and "timed out" in timed_out[0].error
    assert any(s.status is StepStatus.TIMEOUT for s in tracer.tree())


def test_dispatch_returns_one_result_per_call_in_plan_order():
    plan = Plan(calls=[
        ToolCall(tool_name=ToolName.WEB_SEARCH, query=["q"]),
        ToolCall(tool_name=ToolName.RAG, query=["q"]),
    ])
    p = pipeline({
        ToolName.RAG: StubTool(ToolName.RAG, [chunk("r")], delay=0.05),
        ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, [chunk("w")]),
    })
    results = asyncio.run(p.executer.run(plan, context=context(), allowed_tools=frozenset(ToolName), observer=Tracer()))
    assert [r.tool_name for r in results] == [ToolName.WEB_SEARCH, ToolName.RAG]


def test_dispatch_of_an_empty_plan_does_nothing():
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG)})
    assert asyncio.run(p.executer.run(Plan(), context=context(), allowed_tools=frozenset(ToolName), observer=Tracer())) == []


# ===================================================================== merge
def test_merge_deduplicates_by_id_keeping_first_seen_order():
    """The same chunk found by RAG and by GraphRAG must cite once, not twice --
    duplicate citation ids were a real defect."""
    results = [
        ToolResult(tool_name=ToolName.RAG, chunks=[chunk("a"), chunk("b")]),
        ToolResult(tool_name=ToolName.GRAPHRAG, chunks=[chunk("b"), chunk("c")]),
    ]
    assert [c.id for c in gather_results(results)] == ["a", "b", "c"]


def test_merge_of_nothing_is_empty():
    assert gather_results([]) == []


# ======================================================================= run
def test_run_fills_in_the_whole_query_state():
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")])})
    run = asyncio.run(p.run(state("how wide?")))
    assert run.intent is not None
    assert [c.tool_name for c in run.plan.calls] == [ToolName.RAG]
    assert [c.id for c in run.ranked] == ["a"]
    assert run.evaluation is not None
    assert run.round == 0, "no reflection when the context is sufficient"


def test_pipeline_uses_the_evaluators_verdict_before_calling_reflection():
    class SufficientEvaluator(StubEvaluator):
        def is_sufficient(self, evaluation):
            return True

        def reflection(self, *args, **kwargs):
            raise AssertionError("a sufficient evaluation must not be reflected")

    p = pipeline(
        {ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")])},
        evaluator=SufficientEvaluator(score=0.0),
    )

    run = asyncio.run(p.run(state()))
    assert run.round == 0


def test_run_does_a_second_round_only_when_the_evaluator_asks():
    """The evaluator hands back refined queries; the round repeats from intent,
    and the second round's chunks are added to the first round's."""
    rag_tool = StubTool(ToolName.RAG, [chunk("a")])
    p = pipeline(
        {ToolName.RAG: rag_tool,
         ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, [chunk("b")])},
        intent=StubIntent(["kb(all)", "web_search"]),
        evaluator=StubEvaluator(score=0.2, follow_up=["what is missing"]),
    )
    run = asyncio.run(p.run(state()))
    assert run.round == 1
    assert sorted(c.id for c in run.ranked) == ["a", "b"]
    assert len(rag_tool.contexts) == 2
    # Equality, not identity: run.context is derived from the job on each
    # read rather than stored, so it cannot drift from the ticket, and every
    # tool in the turn sees the same three values.
    assert all(tool_context == run.context for tool_context in rag_tool.contexts)


def test_run_stops_at_max_rounds_even_if_the_evaluator_keeps_asking():
    """A reflection that always finds a gap must not spin."""
    from visionagent.config.settings import settings

    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")])},
                 evaluator=StubEvaluator(score=0.0, follow_up=["again"]))
    run = asyncio.run(p.run(state()))
    assert run.round == settings.max_rounds - 1


def test_run_does_not_reflect_when_the_evaluator_returns_nothing():
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")])},
                 evaluator=StubEvaluator(score=0.2, follow_up=None))
    run = asyncio.run(p.run(state()))
    assert run.round == 0


def test_forcing_web_search_adds_it_to_the_plan():
    """The enabled toggle requires a search for this turn."""
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")]),
                  ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, [chunk("w")])})
    run = asyncio.run(p.run(state(web_search=WebSearchMode.FORCE)))
    assert ToolName.WEB_SEARCH in [c.tool_name for c in run.plan.calls]


def test_the_reviewing_step_records_the_verdict():
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")])},
                 evaluator=StubEvaluator(score=0.2, follow_up=None))
    tracer = Tracer()
    asyncio.run(p.run(state(), tracer=tracer))
    reviewing = next(s for s in tracer.tree() if s.label == "reviewing")
    assert reviewing.note == "needs improvement"


def test_the_trace_names_every_stage_of_the_run():
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")])})
    tracer = Tracer()
    asyncio.run(p.run(state(), tracer=tracer))
    assert [s.label for s in tracer.tree()] == [
        "understanding intent", "action planning", "initial search",
        "searching user knowledge base", "result gathering and reranking", "reviewing",
    ]


# ==================================================================== stream
def test_stream_yields_steps_as_they_complete_then_the_run():
    """The route turns these into SSE frames. The pipeline knows nothing about
    SSE, and the route knows nothing about retrieval."""
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")], delay=0.02)})

    async def collect():
        out = []
        async for kind, payload in p.stream(job()):
            out.append((kind, payload))
        return out

    events = asyncio.run(collect())
    kinds = [k for k, _ in events]

    assert kinds[-1] == "run", "the completed run is always last"
    assert "step" in kinds
    final = events[-1][1]
    assert [c.id for c in final.ranked] == ["a"]


def test_answer_slot_receives_typed_chunks_and_separate_citation_ids():
    answer = StubAnswer()
    p = pipeline(
        {ToolName.RAG: StubTool(ToolName.RAG, [chunk("a", "typed evidence")])},
        answer=answer,
    )

    async def collect():
        return [item async for item in p.stream(job())]

    asyncio.run(collect())

    assert answer.professional_contexts == [[
        RetrievedChunk(
            id="a",
            content="typed evidence",
            source_type=SourceType.KNOWLEDGE_BASE,
            score=1.0,
        )
    ]]
    assert answer.citation_ids[0]["a"].startswith("knowledge_base_")


def test_stream_reports_a_step_twice_open_then_closed():
    """A step frame arrives when the step opens, and again when it closes with
    its duration -- which is what makes the tree fill in live."""
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a")])})

    async def collect():
        return [
            (k, v) async for k, v in p.stream(job())
        ]

    steps = [v for k, v in asyncio.run(collect()) if k == "step"]
    by_id: dict[str, list] = {}
    for s in steps:
        by_id.setdefault(s.id, []).append(s.status)

    first_id = steps[0].id
    assert len(by_id[first_id]) == 2, "expected an open and a close frame"


def test_stream_surfaces_evidence_rows():
    p = pipeline({ToolName.RAG: StubTool(ToolName.RAG, [chunk("a", "body")])})

    async def collect():
        return [
            (k, v) async for k, v in p.stream(job())
        ]

    kinds = [k for k, _ in asyncio.run(collect())]
    assert "evidence" in kinds, "a retrieved chunk must appear as evidence"


def test_two_interleaved_turns_on_the_shared_pipeline_keep_their_own_identity():
    """Interleaved turns retain their own immutable request identity."""
    async def go():
        first_entered = asyncio.Event()
        second_entered = asyncio.Event()
        observed: dict[str, tuple[str, str, str]] = {}

        class InterleavingTool:
            async def __call__(
                self, query: str, *, context: ToolContext, emit=None
            ) -> ToolResult:
                if query == "question-a":
                    first_entered.set()
                    await second_entered.wait()
                else:
                    await first_entered.wait()
                    second_entered.set()
                await asyncio.sleep(0)
                observed[query] = (
                    context.run_id,
                    context.user_id,
                    context.session_id,
                )
                return ToolResult(tool_name=ToolName.RAG, chunks=[chunk(query)])

        shared = pipeline({ToolName.RAG: InterleavingTool()})
        await asyncio.gather(
            shared.run(state("question-a", run_id="run-a", user_id="653", session_id="sessA")),
            shared.run(state("question-b", run_id="run-b", user_id="999", session_id="sessB")),
        )
        return observed

    assert asyncio.run(go()) == {
        "question-a": ("run-a", "653", "sessA"),
        "question-b": ("run-b", "999", "sessB"),
    }


def test_casual_forced_web_search_receives_the_turn_context():
    web = StubTool(ToolName.WEB_SEARCH, [chunk("web")])
    p = pipeline(
        {ToolName.WEB_SEARCH: web},
        intent=StubIntent([], scenario="casual"),
    )
    turn = job("hello", web_search=WebSearchMode.FORCE)

    async def collect():
        return [item async for item in p.stream(turn)]

    asyncio.run(collect())
    assert web.contexts == [turn.context]
    assert web.contexts[0] == turn.context


def test_concurrent_children_share_a_parent_rather_than_chaining():
    """Concurrent tool steps are siblings under the complementary-search step."""
    plan = Plan(calls=[
        ToolCall(tool_name=ToolName.RAG, query=["q"]),
        ToolCall(tool_name=ToolName.GRAPHRAG, query=["q"]),
        ToolCall(tool_name=ToolName.WEB_SEARCH, query=["q"]),
    ])
    p = pipeline({
        ToolName.RAG: StubTool(ToolName.RAG, delay=0.05),
        ToolName.GRAPHRAG: StubTool(ToolName.GRAPHRAG, delay=0.02),
        ToolName.WEB_SEARCH: StubTool(ToolName.WEB_SEARCH, delay=0.08),
    })
    tracer = Tracer()

    async def go():
        with tracer.step("complementary search") as parent:
            await p.executer.run(plan, context=context(), allowed_tools=frozenset(ToolName), observer=tracer)
            return parent.id

    parent_id = asyncio.run(go())
    children = [s for s in tracer.tree() if s.id != parent_id]
    assert {s.parent_id for s in children} == {parent_id}, (
        "concurrent tool steps must all hang off the dispatching step: "
        f"{[(s.label, s.parent_id) for s in children]}"
    )
