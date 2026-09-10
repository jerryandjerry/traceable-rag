"""Query orchestration: scenario -> (casual answer) | (intent -> plan ->
dispatch -> rerank -> evaluate -> reflect) -> answer.

The whole turn. The route hands over an immutable QueryJob; the pipeline builds
AgentState and the route encodes what comes back. Every step in between --
including which path the turn takes and the
final answer synthesis -- happens here, so the API imports the pipeline and
nothing below it.

Orchestration only. Every step calls a Protocol; nothing here knows which
provider is behind one, and there is no `if provider == ...` anywhere.

Tools run concurrently with a per-tool timeout. Without that bound, one hung
provider would hold the whole round for its socket timeout with no progress.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from visionagent.config.settings import settings
from visionagent.database.knowledgebase_operations import get_user_history_questions
from visionagent.models import (
    AgentState,
    Answer,
    Evidence,
    Plan,
    QueryJob,
    RetrievedChunk,
    SourceType,
    StepStatus,
    ToolCall,
    ToolName,
    TraceStep,
    WebSearchMode,
)
from visionagent.pipeline.trace import Tracer
from visionagent.service.answer import AnswerGenerator, build_answer_generator
from visionagent.service.evaluator import Evaluator, build_evaluator
from visionagent.service.executer import Executer, build_executer, gather_results
from visionagent.service.intent import IntentParser, build_intent_parser
from visionagent.service.planner import Planner, build_planner
from visionagent.service.rerank import Reranker, build_reranker, rerank_results
from visionagent.service.runtime import close_runtime
from visionagent.utils.prompt import DirectAnswerPrompt

logger = logging.getLogger(__name__)

# A tool that has not answered by now is not going to save the round.
_LABELS = {
    ToolName.RAG: "searching user knowledge base",
    ToolName.GRAPHRAG: "searching user graph base",
    ToolName.WEB_SEARCH: "searching online",
    ToolName.LLM: "searching added context",
}


class QueryPipeline:
    """One question, from intent through to ranked context."""

    def __init__(
        self,
        *,
        intent: IntentParser | None = None,
        planner: Planner | None = None,
        executer: Executer | None = None,
        evaluator: Evaluator | None = None,
        answer: AnswerGenerator | None = None,
        reranker: Reranker | None = None,
        tool_timeout_s: float | None = None,
    ) -> None:
        self.intent = intent or build_intent_parser()
        self.planner = planner or build_planner()
        # The timeout bounds one tool call, which the executer makes, so it is
        # the executer's setting. Kept on this signature because callers and
        # tests already pass it here.
        self.executer = executer or build_executer(tool_timeout_s=tool_timeout_s)
        self.evaluator = evaluator or build_evaluator()
        self.answer = answer or build_answer_generator()
        self.reranker = reranker or build_reranker()

    async def aclose(self) -> None:
        """Release provider resources owned by this application pipeline."""
        await close_runtime()

    def available(self, job: QueryJob | None = None) -> list[ToolName]:
        """Tools that are both installed and authorized for this turn.

        Intersected here rather than checked at dispatch, so the planner never
        proposes a tool it is not allowed to run. Enforcement still happens at
        dispatch as well: a planner defect must not be able to reach a tool
        policy refused.
        """
        names = list(self.executer.names())
        if job is None:
            return names
        return [n for n in names if n in job.authorization.allowed_tools]

    async def run(self, run: AgentState, *, tracer: Tracer | None = None,
                  top_n: int | None = None) -> AgentState:
        """Fill in the run: intent, plan, results, ranked context, evaluation."""
        tracer = tracer or Tracer()
        top_n = settings.rerank_top_n if top_n is None else top_n

        # One turn is a loop: plan, search, rerank, judge -- and if the context
        # is not good enough, the evaluator hands back refined queries and the
        # round repeats from intent. MAX_ROUNDS (counting the first) bounds it.
        queries = [run.question]
        while True:
            # Query-time LLM slots are native async operations. Cancellation
            # reaches their HTTP request or subprocess instead of merely
            # abandoning a blocking call in a worker thread.
            with tracer.step("understanding intent"):
                run.intent = await self.intent.analyze_query_intent(queries)

            with tracer.step("action planning"):
                run.plan = await self.planner.agent_plan(
                    queries,
                    run.intent,
                    session_id=run.session_id,
                    available=self.available(run.job),
                    # Already resolved against policy when the job was minted:
                    # FORCE schedules web search, DISABLED forbids it, AUTO
                    # leaves the decision to intent.
                    web_search=run.job.effective.web_search,
                )

            label = "initial search" if run.round == 0 else "complementary search"
            with tracer.step(label):
                # Results accumulate: a second round adds evidence to the first
                # rather than replacing it.
                run.tool_results.extend(
                    await self.executer.run(
                        run.plan,
                        context=run.context,
                        allowed_tools=run.job.authorization.allowed_tools,
                        observer=tracer,
                    )
                )

            with tracer.step("result gathering and reranking"):
                merged = gather_results(run.tool_results)
                run.ranked = (
                    await rerank_results(
                        merged, run.question, top_n, reranker=self.reranker
                    )
                    if merged else []
                )

            with tracer.step("reviewing") as step:
                run.evaluation = await self.evaluator.evaluate_context_sufficiency(
                    run.ranked, run.question
                )
                sufficient = self.evaluator.is_sufficient(run.evaluation)
                step.note = "looks ok" if sufficient else "needs improvement"

            if sufficient:
                break

            if run.round + 1 >= settings.max_rounds:
                break

            refined = await self.evaluator.reflection(
                run.question, run.ranked, run.evaluation,
                web_search=run.job.effective.web_search,
                available=self.available(run.job),
            )
            if not refined:
                break

            queries = refined
            run.round += 1

        return run


    async def stream(self, job: QueryJob, *, top_n: int | None = None
                     ) -> AsyncIterator[tuple[str, TraceStep | Evidence | AgentState | str]]:
        """The whole turn, yielding `(kind, payload)` as it happens.

        `kind` is "step", "evidence" or "frame" -- "frame" payloads are the
        answer slot's own output, forwarded opaquely -- and the final yield is
        ("run", AgentState). The caller turns these into SSE bytes: the
        pipeline does not know about SSE, and the route does not know about
        retrieval or which path the turn took.
        """
        # The state is built here, from the ticket the API issued. The route
        # never constructs it: orchestration state is what a pipeline is, and a
        # caller that assembles it is deciding how the turn runs.
        run = AgentState(job=job)

        # One deadline over the whole turn: classification, the retrieval
        # loop and the answer. Per-tool timeouts bound each call, and two
        # rounds of four tools plus a long answer add up to more than any
        # client waits. A bound on the loop alone left the casual path and
        # the answer stage open-ended.
        async with asyncio.timeout(settings.turn_timeout_s):
            scenario = await self.intent.analyze_chat_scenario(run.question)
            logger.info(
                "chat scenario selected",
                extra={"scenario": str(scenario)},
            )

            if str(scenario).startswith("casual"):
                async for item in self._casual(run, scenario=str(scenario)):
                    yield item
                yield ("run", run)
                return

            pending: list[tuple[str, TraceStep | Evidence | AgentState]] = []
            seen_evidence = 0

            tracer = Tracer(on_change=lambda step: pending.append(("step", step)))
            # The retrieval loop runs as a task so its progress can be streamed
            # while it works. The task is owned here: the finally below cancels
            # and awaits it when this generator is closed early -- the client
            # disconnected -- so a turn nobody is reading stops calling
            # providers instead of running to completion in the background.
            task = asyncio.create_task(self.run(run, tracer=tracer, top_n=top_n))
            try:
                while not task.done() or pending or seen_evidence < len(tracer.evidence):
                    while pending:
                        yield pending.pop(0)
                    while seen_evidence < len(tracer.evidence):
                        yield ("evidence", tracer.evidence[seen_evidence])
                        seen_evidence += 1
                    if task.done():
                        break
                    await asyncio.sleep(0.02)

                await task
                async for item in self._answer(run, tracer):
                    yield item
                yield ("run", run)
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    async def _casual(self, run: AgentState, *, scenario: str = "casual"
                      ) -> AsyncIterator[tuple[str, TraceStep | str]]:
        """The casual path: no retrieval loop, straight to the answer slot.

        Web search runs when the turn forces it, or when the mode is AUTO and
        the classifier said the question needs current information
        ("casual_web").
        """
        global_context: list[RetrievedChunk] = []
        context_related_questions: list[str] = []
        mode = run.job.effective.web_search
        search = mode is WebSearchMode.FORCE or (
            mode is WebSearchMode.AUTO and scenario == "casual_web"
        )

        if search:
            # The casual path really does search the web when asked; showing
            # the step is what keeps the seconds it takes from looking like a
            # hang.
            step = TraceStep(id="web", label=_LABELS[ToolName.WEB_SEARCH],
                             status=StepStatus.RUNNING, started_at=0.0)
            # Snapshots, as the tracer sends: the consumer may queue this
            # and read it after the mutation below.
            yield ("step", step.model_copy(deep=True))
            started = time.perf_counter()

            # Route every tool call through the same timeout, authorization,
            # and error-handling gateway.
            results = await self.executer.run(
                Plan(calls=[ToolCall(tool_name=ToolName.WEB_SEARCH,
                                     query=[run.question])]),
                context=run.context,
                allowed_tools=run.job.authorization.allowed_tools,
            )
            web_result = results[0] if results else None
            if web_result is None or web_result.error:
                if web_result is not None:
                    logger.error("casual-answer web search failed")
                step.status = StepStatus.FAILED
            else:
                global_context = list(web_result.chunks)
                context_related_questions = list(web_result.related_questions)
                step.status = StepStatus.DONE
            step.duration_s = round(time.perf_counter() - started, 4)
            yield ("step", step.model_copy(deep=True))

        history_questions = await asyncio.to_thread(get_user_history_questions, run.session_id)
        # Keep provider/storage field names out of orchestration. Casual web
        # context has no citation payload, so render the typed chunks directly
        # for the prompt; the answer slot serializes them only at persistence.
        prompt_context = [
            f"[{chunk.doc_name or chunk.url or chunk.id}] {chunk.content}"
            for chunk in global_context
        ]
        final_prompt = DirectAnswerPrompt % (
            prompt_context,
            history_questions,
            run.question,
        )

        frames = self.answer.casual_chat_completion(
            run.session_id,
            run.question,
            run.user_id,
            final_prompt,
            global_context,
            run_id=run.run_id,
            related_questions=context_related_questions,
        )
        collected = _AnswerCollector()
        async for frame in frames:
            collected.absorb(frame)
            # Final before the terminal frame goes out: a client that closes
            # on [DONE] never resumes this generator, so anything after the
            # last yield may not run.
            if "event: end" in frame:
                run.answer = collected.answer(references=run.ranked)
            yield ("frame", frame)
        if run.answer is None:
            run.answer = collected.answer(references=run.ranked)

    async def _answer(self, run: AgentState, tracer: Tracer
                      ) -> AsyncIterator[tuple[str, TraceStep | str]]:
        """The answer step: assemble the ranked context and stream the slot."""
        final_context_list = list(run.ranked)
        context_related_questions = [
            question
            for result in run.tool_results
            for question in result.related_questions
        ]
        history_questions = await asyncio.to_thread(get_user_history_questions, run.session_id)

        # Globally unique citation ids -- the answer cites [doc][cite_<id>]
        # and the frontend resolves them against these.
        timestamp = int(time.time() * 1000) % 1000000
        citation_ids: dict[str, str] = {}
        for i, chunk in enumerate(final_context_list):
            citation_ids[chunk.id] = (
                f"{chunk.source_type.value}_{timestamp}_{i + 1:03d}"
            )
            logger.debug(
                "answer context assigned citation_id=%s chunk_id=%s",
                citation_ids[chunk.id],
                chunk.id,
            )

        final_reference = [
            f"[{citation_ids[chunk.id]}]{chunk.content}"
            for chunk in final_context_list
        ]

        # The user's documents were searched. Say so even when nothing came
        # back, because an empty reference block reads to the model as "no
        # document exists" -- and it then tells the user none was provided,
        # which is false and undermines every answer they have had.
        if not any(
            chunk.source_type is SourceType.KNOWLEDGE_BASE
            for chunk in final_context_list
        ):
            final_reference.insert(0, (
                "[note] The user's own documents were searched and nothing in "
                "them matched this question. Say that their documents do not "
                "cover it. Never say that no document was provided."
            ))
        else:
            # Chunks reached the context but may still not answer the
            # question. Without this, the model admitted the gap in
            # denial-shaped words -- "no file has been shared or attached
            # in this session that specifies..." -- which is grammatically
            # true and reads as though nothing was ever uploaded.
            final_reference.insert(0, (
                "[note] These excerpts are from the user's own uploaded "
                "documents. If they do not answer the question, say that "
                "their documents do not cover it. Never say that no "
                "document or file was provided, shared, or attached."
            ))

        final_prompt = DirectAnswerPrompt % (final_reference, history_questions,
                                             run.question)

        # Related questions are tool-result metadata, not chunk fields. Keep
        # them separate from the typed evidence; the answer boundary preserves
        # their legacy nesting and deduplicates the top-level frame.
        web_chunks = [
            chunk
            for chunk in final_context_list
            if chunk.source_type is SourceType.WEB_SEARCH
        ]
        top_snippets = web_chunks

        answering = TraceStep(id="answer", label="generating answer",
                              status=StepStatus.RUNNING,
                              started_at=round(tracer.elapsed, 4))
        yield ("step", answering.model_copy(deep=True))

        # Media from this turn's web search, if one ran. It is the tool's
        # output, so it exists only under the same authorization as the
        # search; the answer slot never reaches a provider on its own.
        media: dict[str, list[dict[str, Any]]] = {"images": [], "videos": []}
        for r in run.tool_results:
            if r.tool_name is ToolName.WEB_SEARCH and not r.error:
                media["images"].extend(r.images)
                media["videos"].extend(r.videos)

        start_answer_time = time.time()
        frames = self.answer.get_chat_completion(
            run.session_id, run.question, final_context_list, run.user_id,
            final_prompt, context_related_questions, top_snippets,
            run_id=run.run_id, citation_ids=citation_ids, media=media,
        )
        collected = _AnswerCollector()
        async for frame in frames:
            collected.absorb(frame)
            # Close the step and final the state BEFORE forwarding the end
            # frame: the client stops reading at [DONE], so a step frame after
            # it is never seen, and this generator may never be resumed.
            if "event: end" in frame:
                answering.status = StepStatus.DONE
                answering.duration_s = round(time.time() - start_answer_time, 4)
                run.answer = collected.answer(references=run.ranked)
                yield ("step", answering.model_copy(deep=True))
            yield ("frame", frame)
        if run.answer is None:
            run.answer = collected.answer(references=run.ranked)


class _AnswerCollector:
    """Build the typed answer from the answer slot's SSE wire frames.

    Parsing at the streaming boundary keeps ``AgentState.answer`` complete
    without changing the slot's public wire contract.
    """

    def __init__(self) -> None:
        self.content: list[str] = []
        self.think: list[str] = []
        self.related_questions: list[str] = []
        self.images: list[str] = []
        self.videos: list[str] = []

    def absorb(self, frame: str) -> None:
        for line in frame.splitlines():
            if not line.startswith("data: "):
                continue
            try:
                payload = json.loads(line[6:])
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            text = payload.get("content")
            if payload.get("role") == "assistant" and isinstance(text, str):
                (self.think if payload.get("thinking") else self.content).append(text)
            for q in payload.get("recommended_questions") or []:
                if isinstance(q, str) and q not in self.related_questions:
                    self.related_questions.append(q)
            for i in (payload.get("image_results") or {}).get("images") or []:
                if isinstance(i, dict) and i.get("imageUrl"):
                    self.images.append(str(i["imageUrl"]))
            for v in (payload.get("video_results") or {}).get("videos") or []:
                if isinstance(v, dict) and v.get("link"):
                    self.videos.append(str(v["link"]))

    def answer(self, *, references: list[Any]) -> Answer:
        return Answer(
            content="".join(self.content),
            think="".join(self.think),
            references=list(references),
            related_questions=list(self.related_questions),
            images=list(self.images),
            videos=list(self.videos),
        )
