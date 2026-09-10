"""Tests for intent/, planner/, evaluator/ and answer/.

Intended function of the intent component:
    parse(query) -> exactly one Intent: CASUAL only when the classifier says
    so, KNOWLEDGE otherwise; intents empty for casual; keywords always present

Intended function of the planner component:
    plan(query, intent, available) -> exactly one ToolCall per matched intent,
    de-duplicated, restricted to tools that exist, and never empty
    force(plan, tool) -> the plan plus that tool, or the plan unchanged if it
    is already scheduled

Intended function of the evaluator component:
    evaluate(query, chunks) -> exactly the model's score/reasons/comments,
    clamped, defaulting to sufficient when the evaluator itself fails
    reflect(...) -> None when sufficient; otherwise exactly the follow-up
    calls, dropping tools that do not exist

Intended function of the answer component:
    build_prompt(query, chunks) -> exactly the citation-marked reference block
"""
from __future__ import annotations

import asyncio
import json

import pytest

from visionagent.models import (
    Evaluation,
    Intent,
    Plan,
    RetrievedChunk,
    Scenario,
    SourceType,
    ToolCall,
    ToolName,
    WebSearchMode,
)
from visionagent.service.answer import AnswerGenerator, build_answer_generator
from visionagent.service.evaluator import Evaluator, build_evaluator
from visionagent.service.intent import IntentParser, build_intent_parser
from visionagent.service.planner import Planner, build_planner
from visionagent.service.planner import rule_based as agent_mod

ALL = [ToolName.RAG.value, ToolName.GRAPHRAG.value,
       ToolName.WEB_SEARCH.value, ToolName.LLM.value]


def _async_reply(value):
    async def reply(*args, **kwargs):
        return value

    return reply


def chunk(cid: str, body: str, src=SourceType.KNOWLEDGE_BASE) -> RetrievedChunk:
    return RetrievedChunk(id=cid, content=body, source_type=src)


# ==================================================================== intent
def test_intent_reports_casual_only_when_the_classifier_says_so(monkeypatch):
    import visionagent.service.intent.llm as agent
    import visionagent.utils.keyword_extraction as kw

    monkeypatch.setattr(agent, "analyze_chat_scenario", _async_reply("casual"))
    monkeypatch.setattr(agent, "middle_json_model", _async_reply('["kb(filter)"]'))
    monkeypatch.setattr(kw, "extract_keywords_advanced", lambda q: (["hi"], ["hello"]))
    got = asyncio.run(build_intent_parser("llm").analyze_query_intent(["hello there"]))
    assert got == Intent(scenario=Scenario.CASUAL, intents=[],
                         keywords_high=["hi"], keywords_low=["hello"])


def test_intent_skips_the_second_llm_call_for_casual_queries(monkeypatch):
    """A greeting must not pay for an intent classification."""
    import visionagent.service.intent.llm as agent
    import visionagent.utils.keyword_extraction as kw

    called: list[str] = []
    monkeypatch.setattr(agent, "analyze_chat_scenario", _async_reply("casual"))
    # The intent-classification LLM call. A greeting must not pay for one, so
    # this must never run -- asserted, not implied by the empty intents list,
    # because an intent list can be empty for other reasons.
    async def model(prompt):
        called.append(prompt)
        return '["kb(filter)"]'

    monkeypatch.setattr(agent, "middle_json_model", model)
    monkeypatch.setattr(kw, "extract_keywords_advanced", lambda q: ([], []))
    got = asyncio.run(build_intent_parser("llm").analyze_query_intent(["hi"]))
    assert called == [], "a casual query paid for an intent classification"
    assert got.scenario is Scenario.CASUAL
    assert got.intents == []


def test_intent_treats_anything_not_casual_as_knowledge(monkeypatch):
    """The safe default: an unrecognised classifier reply must not skip
    retrieval."""
    import visionagent.service.intent.llm as agent
    import visionagent.utils.keyword_extraction as kw

    monkeypatch.setattr(
        agent, "analyze_chat_scenario", _async_reply("something else")
    )
    monkeypatch.setattr(agent, "middle_json_model", _async_reply('["kb(filter)"]'))
    monkeypatch.setattr(kw, "extract_keywords_advanced", lambda q: ([], []))
    got = asyncio.run(build_intent_parser("llm").analyze_query_intent(["q"]))
    assert got.scenario is Scenario.KNOWLEDGE


def test_keyword_only_parser_needs_no_llm(monkeypatch):
    """The cheap end of the scale the Protocol exists to make swappable."""
    import visionagent.service.intent.llm as agent
    import visionagent.utils.keyword_extraction as kw

    def explode(*a, **k):
        raise AssertionError("the keyword parser must not call an LLM")

    monkeypatch.setattr(agent, "analyze_chat_scenario", explode)
    monkeypatch.setattr(kw, "extract_keywords_advanced", lambda q: (["curb"], ["width"]))
    got = asyncio.run(
        build_intent_parser("keywords").analyze_query_intent(["curb width"])
    )
    assert got == Intent(scenario=Scenario.KNOWLEDGE, intents=["kb(filter)"],
                         keywords_high=["curb"], keywords_low=["width"])


def test_intent_protocol_and_factory():
    assert isinstance(build_intent_parser("llm"), IntentParser)
    assert isinstance(build_intent_parser("keywords"), IntentParser)
    with pytest.raises(ValueError, match="unknown intent parser"):
        build_intent_parser("bert")


# =================================================================== planner
def test_a_kb_intent_queues_both_retrievers():
    """One knowledge-base intent schedules both text and graph retrieval."""
    got = asyncio.run(agent_mod.agent_plan(
        ["how wide?"],
        Intent(scenario=Scenario.KNOWLEDGE, intents=["kb(filter)", "web_search"]),
        available=ALL,
    ))
    assert got == Plan(calls=[
        ToolCall(tool_name=ToolName.RAG, query=["how wide?"]),
        ToolCall(tool_name=ToolName.GRAPHRAG, query=["how wide?"]),
        ToolCall(tool_name=ToolName.WEB_SEARCH, query=["how wide?"]),
    ])


@pytest.mark.parametrize("emitted", ["kb(filter)", "kb(all)", "KB(All)", "kb(doc_123)", "kb"])
def test_the_kb_intent_is_matched_however_the_model_spells_the_filter(emitted):
    """Every accepted knowledge-base filter spelling schedules both retrievers."""
    got = asyncio.run(agent_mod.agent_plan(
        ["q"], Intent(scenario=Scenario.KNOWLEDGE, intents=[emitted]), available=ALL
    ))
    assert [c.tool_name for c in got.calls] == [ToolName.RAG, ToolName.GRAPHRAG], emitted


def test_plan_deduplicates_a_repeated_intent():
    got = asyncio.run(agent_mod.agent_plan(
        ["q"],
        Intent(scenario=Scenario.KNOWLEDGE, intents=["kb(filter)", "kb(all)"]),
        available=ALL,
    ))
    assert [c.tool_name for c in got.calls] == [ToolName.RAG, ToolName.GRAPHRAG]


def test_plan_drops_intents_whose_tool_is_not_registered():
    """agent_plan cannot schedule a tool the registry does not have."""
    got = asyncio.run(agent_mod.agent_plan(
        ["q"],
        Intent(scenario=Scenario.KNOWLEDGE, intents=["kb(filter)", "web_search"]),
        available=[ToolName.RAG.value],
    ))
    assert [c.tool_name for c in got.calls] == [ToolName.RAG]


def test_plan_is_never_empty():
    """An empty plan reads downstream as 'every tool found nothing', which is
    a different statement from 'nothing was worth running'."""
    got = asyncio.run(agent_mod.agent_plan(
        ["q"], Intent(scenario=Scenario.KNOWLEDGE, intents=[]), available=ALL
    ))
    assert [c.tool_name for c in got.calls] == [ToolName.LLM]


def test_plan_of_unknown_intents_falls_back_to_the_model():
    got = asyncio.run(agent_mod.agent_plan(
        ["q"], Intent(scenario=Scenario.KNOWLEDGE, intents=["telepathy"]), available=ALL
    ))
    assert [c.tool_name for c in got.calls] == [ToolName.LLM]


def test_web_search_adds_a_tool_the_user_asked_for():
    """Web toggle ON forces a search for this turn, whatever the intents said."""
    got = asyncio.run(agent_mod.agent_plan(
        ["q"], Intent(scenario=Scenario.KNOWLEDGE, intents=["kb(all)"]),
        available=ALL, web_search=WebSearchMode.FORCE,
    ))
    assert [c.tool_name for c in got.calls] == [
        ToolName.RAG, ToolName.GRAPHRAG, ToolName.WEB_SEARCH]


def test_web_search_is_idempotent_when_already_scheduled():
    got = asyncio.run(agent_mod.agent_plan(
        ["q"], Intent(scenario=Scenario.KNOWLEDGE, intents=["web_search"]),
        available=ALL, web_search=WebSearchMode.FORCE,
    ))
    assert [c.tool_name for c in got.calls] == [ToolName.WEB_SEARCH]


def test_planner_protocol_and_factory():
    assert isinstance(build_planner(), Planner)
    with pytest.raises(ValueError, match="unknown planner"):
        build_planner("llm")


# ================================================================= evaluator
# Exercise the exact evaluator implementation returned by the slot factory.
def _fake_model(monkeypatch, reply: str):
    import visionagent.service.evaluator.llm as agent

    monkeypatch.setattr(agent, "middle_json_model", _async_reply(reply))
    return agent


def test_evaluate_returns_exactly_the_models_verdict(monkeypatch):
    agent = _fake_model(monkeypatch, json.dumps(
        {"sufficient_score": 0.42, "reasons": "thin", "comments": "need widths"}))
    assert asyncio.run(agent.evaluate_context_sufficiency(
        [chunk("c1", "x")], "q"
    )) == Evaluation(
        sufficient_score=0.42, reasons="thin", comments="need widths"
    )


def test_evaluate_reports_insufficient_when_there_is_no_context_at_all():
    import visionagent.service.evaluator.llm as agent

    got = asyncio.run(agent.evaluate_context_sufficiency([], "q"))
    assert got.sufficient_score == 0.0 and not agent.is_sufficient(got)


def test_evaluate_defaults_to_sufficient_when_the_evaluator_itself_fails(monkeypatch):
    """A failed evaluator must not trigger an extra round of retrieval."""
    agent = _fake_model(monkeypatch, "not json at all")
    got = asyncio.run(agent.evaluate_context_sufficiency([chunk("c1", "x")], "q"))
    assert got.sufficient_score == 1.0 and agent.is_sufficient(got)


def test_evaluate_clamps_an_out_of_range_score(monkeypatch):
    agent = _fake_model(monkeypatch, json.dumps(
        {"sufficient_score": 7, "reasons": "", "comments": ""}))
    got = asyncio.run(agent.evaluate_context_sufficiency([chunk("c1", "x")], "q"))
    assert got.sufficient_score == 1.0


def test_reflect_returns_none_when_the_context_is_sufficient(monkeypatch):
    import visionagent.service.evaluator.llm as agent

    async def explode(prompt):
        raise AssertionError("reflection must not call the model on a sufficient context")

    monkeypatch.setattr(agent, "middle_json_model", explode)
    assert asyncio.run(agent.reflection(
        "q", [], Evaluation(sufficient_score=0.9), available=ALL
    )) is None


@pytest.mark.parametrize(
    "score,sufficient",
    [(0.0, False), (0.5, False), (0.51, True), (1.0, True)],
)
def test_evaluator_owns_the_sufficiency_threshold(score, sufficient):
    import visionagent.service.evaluator.llm as agent

    assert agent.is_sufficient(Evaluation(sufficient_score=score)) is sufficient


def test_reflect_returns_the_follow_up_calls(monkeypatch):
    agent = _fake_model(monkeypatch, json.dumps([
        {"intent": "kb(all)", "query": "curb"},
        {"intent": "kb(all)", "query": "width"},
        {"intent": "web_search", "query": "curb width standard"},
    ]))
    got = asyncio.run(agent.reflection(
        "q", [], Evaluation(sufficient_score=0.2),
        web_search=WebSearchMode.AUTO, available=ALL,
    ))
    assert got == ["curb", "width", "curb width standard"]


def test_reflect_matches_the_kb_intent_however_the_model_spells_the_filter(monkeypatch):
    """A concrete knowledge-base filter remains a knowledge-base intent."""
    agent = _fake_model(monkeypatch, json.dumps([{"intent": "kb(all)", "query": "y"}]))
    got = asyncio.run(agent.reflection(
        "q", [], Evaluation(sufficient_score=0.1),
        web_search=WebSearchMode.AUTO, available=ALL,
    ))
    assert got == ["y"]


def test_reflect_returns_what_to_search_not_which_tools(monkeypatch):
    """The reflection prompt can name anything; the plan may not."""
    agent = _fake_model(monkeypatch, json.dumps([
        {"intent": "kb(all)", "query": "y"},
        {"intent": "web_search", "query": "x"},
    ]))
    got = asyncio.run(agent.reflection(
        "q", [], Evaluation(sufficient_score=0.1), web_search=WebSearchMode.AUTO,
        available=[ToolName.RAG.value]))
    # Which tools can run is the planner's business; reflection only says what
    # to search for next.
    assert got == ["y", "x"]


def test_reflect_forces_the_web_when_the_toggle_is_on(monkeypatch):
    agent = _fake_model(monkeypatch, json.dumps([{"intent": "kb(all)", "query": "y"}]))
    got = asyncio.run(agent.reflection(
        "q", [], Evaluation(sufficient_score=0.1),
        web_search=WebSearchMode.FORCE, available=ALL,
    ))
    assert got == ["y", "q"], "a forced web toggle must survive the extra round"


def test_evaluator_protocol_and_factory():
    """The agent module itself satisfies the Protocol -- no adapter class."""
    assert isinstance(build_evaluator(), Evaluator)
    with pytest.raises(ValueError, match="unknown evaluator"):
        build_evaluator("heuristic")


# ==================================================================== answer
# Exercise the answer module that streams frames and persists the completed turn.
def test_the_answer_slot_is_the_module_that_actually_answers():
    """The factory must return the code the route runs, not a second
    implementation of part of it."""
    from visionagent.service.answer import chat

    assert build_answer_generator() is chat


def test_the_pipeline_calls_the_slot_rather_than_importing_chat_directly(app_dir):
    src = (app_dir / "pipeline" / "query.py").read_text(encoding="utf-8")
    assert "build_answer_generator()" in src, (
        "the pipeline bypasses the answer slot, so ANSWER cannot swap it"
    )
    assert "from visionagent.service.answer.chat import" not in src

    # The route is HTTP only: it imports the pipeline and nothing below it.
    route = (app_dir / "api" / "routes" / "ai_search_rt.py").read_text(encoding="utf-8")
    for slot in ("visionagent.service.answer", "visionagent.service.intent", "visionagent.service.executer.tools",
                 "visionagent.service.evaluator", "visionagent.service.planner"):
        assert f"from {slot}" not in route, (
            f"the route imports {slot}; slots are the pipeline's to call"
        )


def test_each_reference_carries_its_citation_id_and_source_kind():
    """The prompt requires [doc][cite_XXX] / [web][cite_XXX], so every
    reference handed to the model has to carry an id that resolves, and the
    citation id itself encodes which kind of source it came from."""
    from visionagent.models import RetrievedChunk, SourceType
    from visionagent.service.answer.chat import _serialize_retrieved_chunks

    rows = _serialize_retrieved_chunks(
        [
            RetrievedChunk(
                id="c1",
                content="Sidewalk widths.",
                source_type=SourceType.KNOWLEDGE_BASE,
                score=0.8,
                doc_name="standards.pdf",
                page_num=4,
                images=["REF64"],
            ),
            RetrievedChunk(
                id="w1",
                content="A blog post.",
                source_type=SourceType.WEB_SEARCH,
                url="https://example.com",
            ),
        ],
        citation_ids={
            "c1": "knowledge_base_123456_001",
            "w1": "web_search_123456_002",
        },
        related_questions=["How wide?"],
    )

    assert rows == [
        {
            "chunk_id": "c1",
            "id": "c1",
            "content_with_weight": "Sidewalk widths.",
            "docnm": "standards.pdf",
            "docnm_kwd": "standards.pdf",
            "doc_id": "",
            "kb_id": "",
            "important_kwd": [],
            "sim": 0.8,
            "similarity": 0.8,
            "ref_images": ["REF64"],
            "source_type": "knowledge_base",
            "page_num": 4,
            "citation_id": "knowledge_base_123456_001",
        },
        {
            "chunk_id": "w1",
            "id": "w1",
            "content_with_weight": "A blog post.",
            "docnm": "",
            "docnm_kwd": "",
            "doc_id": "https://example.com",
            "kb_id": "",
            "important_kwd": [],
            "sim": 0.0,
            "similarity": 0.0,
            "ref_images": [],
            "source_type": "web_search",
            "url": "https://example.com",
            "title": "",
            "related_questions": ["How wide?"],
            "image_urls": [],
            "video_urls": [],
            "citation_id": "web_search_123456_002",
        },
    ]

    reference = [f"[{r['citation_id']}]{r['content_with_weight']}" for r in rows]
    assert reference[0].startswith("[knowledge_base_123456_001]")
    assert reference[1].startswith("[web_search_123456_002]")
    assert "Sidewalk widths." in reference[0]
    assert "A blog post." in reference[1]


def test_answer_serializes_typed_chunks_only_at_wire_and_database_boundary(monkeypatch):
    import visionagent.service.answer.chat as chat

    class LLM:
        async def stream(self, **kwargs):
            yield "answer", None

    persisted = {}

    async def persist(session_id, question, model_answer, documents,
                      related_questions, think, user_id, *, run_id):
        persisted.update(
            documents=documents,
            related_questions=related_questions,
            run_id=run_id,
        )
        return []

    monkeypatch.setattr(chat, "_llm", lambda: LLM())
    monkeypatch.setattr(chat, "_persist", persist)
    evidence = RetrievedChunk(
        id="w1",
        content="A typed web result.",
        source_type=SourceType.WEB_SEARCH,
        doc_name="Example",
        url="https://example.com",
    )

    async def collect():
        return [frame async for frame in chat.get_chat_completion(
            "s1",
            "question",
            [evidence],
            "u1",
            "prompt",
            ["First?", "First?", "Second?"],
            [evidence],
            run_id="answer-run",
            citation_ids={"w1": "web_search_123456_001"},
            media={"images": [], "videos": []},
        )]

    frames = asyncio.run(collect())
    messages = [
        json.loads(frame.split("data: ", 1)[1])
        for frame in frames
        if frame.startswith("event: message")
    ]

    documents = messages[0]["documents"]
    assert messages[1]["web_search"] == documents
    assert messages[3]["recommended_questions"] == ["First?", "Second?"]
    assert messages[6]["citations"] == documents
    assert persisted == {
        "documents": documents,
        "related_questions": ["First?", "Second?"],
        "run_id": "answer-run",
    }
    assert evidence.model_dump().keys() == RetrievedChunk.model_fields.keys()
    assert "citation_id" not in evidence.model_dump()


def test_answer_metadata_prompts_preserve_the_query_language(monkeypatch):
    import visionagent.service.answer.chat as chat

    prompts: list[str] = []

    class LLM:
        async def complete(self, *, prompt, **kwargs):
            prompts.append(prompt)
            if "recommended_questions" in prompt:
                return '{"recommended_questions": ["One?", "Two?", "Three?"]}'
            return '{"session_name": "Name"}'

    monkeypatch.setattr(chat, "_llm", lambda: LLM())
    questions = asyncio.run(chat.generate_recommended_questions("¿Qué ocurrió?", []))
    name = asyncio.run(chat.generate_session_name("¿Qué ocurrió?"))

    assert questions == ["One?", "Two?", "Three?"]
    assert name == "Name"
    assert len(prompts) == 2
    assert all("same language" in prompt for prompt in prompts)
    assert all("¿Qué ocurrió?" in prompt for prompt in prompts)


def test_answer_protocol_and_factory():
    assert isinstance(build_answer_generator(), AnswerGenerator)
    with pytest.raises(ValueError, match="unknown answer generator"):
        build_answer_generator("mapreduce")
