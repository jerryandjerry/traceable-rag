"""Tests for the llm/ slot.

No network. The point of the slot is that a fake implementation can drive
everything downstream, so these tests are also the proof that it can.
"""
from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel, Field

from visionagent.providers.llm import LLM, LLMError, build_llm
from visionagent.providers.llm.dashscope import extract_json


class Verdict(BaseModel):
    ok: bool
    score: float = Field(ge=0.0, le=1.0)


class FakeLLM:
    """A complete LLM with no provider behind it."""

    name = "fake"

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def complete(self, *, prompt, system=None, model=None, temperature=0.0,
                       json_object=False) -> str:
        self.calls.append({"prompt": prompt, "system": system, "model": model,
                           "json_object": json_object})
        return self.reply

    async def stream(self, *, prompt, system=None, model=None,
                     temperature=0.0) -> AsyncIterator[tuple[str, str]]:
        for word in self.reply.split():
            yield (word + " ", "")

    async def complete_json(
        self, *, prompt, schema, system=None, model=None, temperature=0.0
    ):
        import json as _json
        return schema.model_validate(_json.loads(extract_json(self.reply)))

    async def aclose(self) -> None:
        return None


def run(awaitable):
    return asyncio.run(awaitable)


async def collect(stream):
    return [item async for item in stream]


# ------------------------------------------------------------------ protocol
def test_real_implementation_satisfies_the_protocol():
    assert isinstance(build_llm(), LLM)


def test_a_fake_satisfies_the_protocol():
    """If this fails, downstream components cannot be tested without a provider."""
    assert isinstance(FakeLLM(), LLM)


def test_provider_contract_is_native_async():
    """A regression to sync clients would make turn cancellation ineffective."""
    from visionagent.providers.llm.claude_cli import ClaudeCLI
    from visionagent.providers.llm.dashscope import DashScopeLLM

    for provider in (DashScopeLLM, ClaudeCLI):
        assert inspect.iscoroutinefunction(provider.complete)
        assert inspect.iscoroutinefunction(provider.complete_json)
        assert inspect.iscoroutinefunction(provider.aclose)
        assert inspect.isasyncgenfunction(provider.stream)


def test_factory_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        build_llm("gpt-9")


def test_factory_reads_the_env_selector(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    assert build_llm().name == "dashscope"


# --------------------------------------------------------------- json extraction
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"ok": true}', '{"ok": true}'),
        ('```json\n{"ok": true}\n```', '{"ok": true}'),
        ('```\n{"ok": true}\n```', '{"ok": true}'),
        ('Here you go: {"ok": true} hope that helps', '{"ok": true}'),
        ('["kb(filter)", "web_search"]', '["kb(filter)", "web_search"]'),
        ('text before ["a"] text after', '["a"]'),
    ],
)
def test_extract_json_tolerates_llm_wrapping(raw: str, expected: str):
    """One parser accepts the common wrappers emitted by model providers."""
    assert extract_json(raw) == expected


def test_extract_json_rejects_an_empty_response():
    with pytest.raises(LLMError):
        extract_json("")


# ------------------------------------------------------------- complete_json
def test_complete_json_returns_a_validated_model():
    llm = FakeLLM('```json\n{"ok": true, "score": 0.8}\n```')
    v = run(llm.complete_json(prompt="p", schema=Verdict))
    assert isinstance(v, Verdict) and v.ok and v.score == 0.8


def test_complete_json_rejects_out_of_range_values():
    """The contract's bounds apply to model output, not just to our own code."""
    from pydantic import ValidationError

    llm = FakeLLM('{"ok": true, "score": 5}')
    with pytest.raises(ValidationError):
        run(llm.complete_json(prompt="p", schema=Verdict))


def test_dashscope_complete_json_raises_llmerror_not_keyerror(monkeypatch):
    """A malformed response must fail at the LLM boundary naming the schema,
    not three frames later as a missing dict key."""
    from visionagent.providers.llm.dashscope import DashScopeLLM

    llm = DashScopeLLM()
    async def complete(**kw):
        return "not json at all"

    monkeypatch.setattr(llm, "complete", complete)
    with pytest.raises(LLMError, match="unparseable JSON for Verdict"):
        run(llm.complete_json(prompt="p", schema=Verdict))


def test_dashscope_complete_json_reports_schema_violations(monkeypatch):
    from visionagent.providers.llm.dashscope import DashScopeLLM

    llm = DashScopeLLM()
    async def complete(**kw):
        return '{"ok": true}'

    monkeypatch.setattr(llm, "complete", complete)
    with pytest.raises(LLMError, match="not a valid Verdict"):
        run(llm.complete_json(prompt="p", schema=Verdict))


# ------------------------------------------------------------------ swapping
def test_a_component_can_be_driven_by_a_fake(monkeypatch):
    """middle_json_model goes through the slot, so it needs no network.

    _LLM is the module-level cache inside providers/llm/helpers.py; patching the
    package would set an attribute the function never reads."""
    import visionagent.providers.llm as agent
    import visionagent.providers.llm.helpers as helpers

    fake = FakeLLM('{"tools": ["RAG"]}')
    monkeypatch.setattr(helpers, "_LLM", fake)
    assert run(agent.middle_json_model("pick tools")) == '{"tools": ["RAG"]}'
    assert fake.calls[0]["json_object"] is True


def test_shared_application_llm_is_closed_and_cleared(monkeypatch):
    """The lifespan can release the one client bound to its event loop."""
    import visionagent.providers.llm.helpers as helpers

    fake = FakeLLM()
    close_calls = 0

    async def close() -> None:
        nonlocal close_calls
        close_calls += 1

    monkeypatch.setattr(fake, "aclose", close)
    monkeypatch.setattr(helpers, "_LLM", fake)

    run(helpers.close_llm())
    assert close_calls == 1
    assert helpers._LLM is None

    # Shutdown is idempotent when startup never constructed a client or the
    # cleanup path is called again.
    run(helpers.close_llm())
    assert close_calls == 1


def test_streaming_yields_content_and_reasoning_channels():
    """Reasoning models return reasoning_content separately and the frontend
    renders it in its own block, so the pair is part of the contract."""
    deltas = run(collect(FakeLLM("one two").stream(prompt="p")))
    assert deltas == [("one ", ""), ("two ", "")]


# ==========================================================================
# EXACT BEHAVIOUR
#
# Intended function of the LLM component:
#   complete()      prompt -> exactly the provider's assistant text, unchanged
#   stream()        prompt -> exactly the provider's deltas, as (content,
#                   reasoning) pairs, in order, empties preserved as ""
#   complete_json() prompt -> exactly the model the schema describes, built
#                   from the JSON embedded anywhere in the response
#
# Every expectation below is an exact value, not a property.
# ==========================================================================
class StubResponse:
    """Shapes a fake OpenAI SDK response so the real DashScopeLLM can be driven."""

    def __init__(self, text: str) -> None:
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class StubDelta:
    def __init__(self, content: str | None, reasoning: str | None = None) -> None:
        self.content = content
        self.reasoning_content = reasoning


class StubChunk:
    def __init__(self, delta: StubDelta | None) -> None:
        self.choices = [type("C", (), {"delta": delta})()] if delta else []


class StubAsyncStream:
    def __init__(self, items) -> None:
        self._items = iter(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.closed = True


def _stub_client(llm, result):
    """Replace the SDK call on a real DashScopeLLM instance."""
    async def create(**kw):
        return StubAsyncStream(result) if isinstance(result, list) else result

    llm._client = type(
        "Client", (),
        {"chat": type("Chat", (), {"completions": type(
            "Comp", (), {"create": staticmethod(create)})()})()},
    )()
    return llm


def test_complete_returns_the_assistant_text_verbatim():
    from visionagent.providers.llm.dashscope import DashScopeLLM

    llm = _stub_client(DashScopeLLM(), StubResponse("Curb extensions are 1.8m."))
    assert run(llm.complete(prompt="how wide?")) == "Curb extensions are 1.8m."


def test_complete_returns_empty_string_not_none_when_the_model_says_nothing():
    """Callers concatenate the result; None would raise on the first +=."""
    from visionagent.providers.llm.dashscope import DashScopeLLM

    llm = _stub_client(DashScopeLLM(), StubResponse(None))
    assert run(llm.complete(prompt="x")) == ""


def test_stream_yields_exactly_the_deltas_in_order():
    from visionagent.providers.llm.dashscope import DashScopeLLM

    chunks = [
        StubChunk(StubDelta("Curb ")),
        StubChunk(StubDelta("extensions", "because the query asks about width")),
        StubChunk(StubDelta(None)),          # keepalive: no content
        StubChunk(None),                     # no choices at all
        StubChunk(StubDelta(" are 1.8m.")),
    ]
    llm = _stub_client(DashScopeLLM(), chunks)
    assert run(collect(llm.stream(prompt="how wide?"))) == [
        ("Curb ", ""),
        ("extensions", "because the query asks about width"),
        ("", ""),
        (" are 1.8m.", ""),
    ]


def test_complete_json_builds_exactly_the_requested_model():
    from visionagent.providers.llm.dashscope import DashScopeLLM

    llm = _stub_client(
        DashScopeLLM(),
        StubResponse('Sure!\n```json\n{"ok": true, "score": 0.42}\n```\nHope that helps.'),
    )
    assert run(llm.complete_json(
        prompt="judge", schema=Verdict
    )) == Verdict(ok=True, score=0.42)


def test_json_mode_is_requested_only_when_asked_for():
    """Three call sites depend on provider JSON mode; sending it always would
    change behaviour for the ones that do not want it."""
    from visionagent.providers.llm.dashscope import DashScopeLLM

    seen: list[dict] = []
    llm = DashScopeLLM()

    async def create(**kw):
        seen.append(kw)
        return StubResponse("{}")

    llm._client = type(
        "Client", (),
        {"chat": type("Chat", (), {"completions": type(
            "Comp", (), {"create": staticmethod(create)})()})()},
    )()
    run(llm.complete(prompt="a"))
    run(llm.complete(prompt="b", json_object=True))
    assert "response_format" not in seen[0]
    assert seen[1]["response_format"] == {"type": "json_object"}


def test_system_prompt_is_placed_before_the_user_message():
    from visionagent.providers.llm.dashscope import DashScopeLLM

    seen: list[dict] = []
    llm = DashScopeLLM()

    async def create(**kw):
        seen.append(kw)
        return StubResponse("")

    llm._client = type(
        "Client", (),
        {"chat": type("Chat", (), {"completions": type(
            "Comp", (), {"create": staticmethod(create)})()})()},
    )()
    run(llm.complete(prompt="the question", system="you are an architect"))
    assert seen[0]["messages"] == [
        {"role": "system", "content": "you are an architect"},
        {"role": "user", "content": "the question"},
    ]


def test_cancelling_dashscope_completion_cancels_the_http_request(monkeypatch):
    from visionagent.providers.llm.dashscope import DashScopeLLM

    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def create(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    llm = DashScopeLLM()
    llm._client = type(
        "Client", (),
        {"chat": type("Chat", (), {"completions": type(
            "Comp", (), {"create": staticmethod(create)})()})()},
    )()

    async def cancel():
        task = asyncio.create_task(llm.complete(prompt="hi"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set()

    run(cancel())


def test_dashscope_stream_closes_its_http_response():
    from visionagent.providers.llm.dashscope import DashScopeLLM

    stream = StubAsyncStream([StubChunk(StubDelta("one"))])

    async def create(**kwargs):
        return stream

    llm = DashScopeLLM()
    llm._client = type(
        "Client", (),
        {"chat": type("Chat", (), {"completions": type(
            "Comp", (), {"create": staticmethod(create)})()})()},
    )()
    assert run(collect(llm.stream(prompt="hi"))) == [("one", "")]
    assert stream.closed


# ======================================================= claude cli transport
class _BytesReader:
    def __init__(self, *, lines=(), body: str = "") -> None:
        self._lines = iter([f"{line}\n".encode() for line in lines])
        self._body = body.encode()

    async def readline(self) -> bytes:
        return next(self._lines, b"")

    async def read(self) -> bytes:
        return self._body


class _Stdin:
    def __init__(self, call: dict) -> None:
        self._call = call

    def write(self, value: bytes) -> None:
        self._call["input"] = value.decode()

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        return None


class _FakeProcess:
    def __init__(self, transport: _Transport, call: dict) -> None:
        self._transport = transport
        self._call = call
        self.returncode: int | None = None
        self.stdin = _Stdin(call)
        self.stdout = _BytesReader(lines=transport.lines)
        self.stderr = _BytesReader(body=transport.stderr)
        self.terminated = False
        self.killed = False

    async def communicate(self, value: bytes) -> tuple[bytes, bytes]:
        self._call["input"] = value.decode()
        self.returncode = self._transport.returncode
        return self._transport.stdout.encode(), self._transport.stderr.encode()

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._transport.returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _Transport:
    def __init__(
        self,
        *,
        stdout: str = "answer",
        stderr: str = "",
        returncode: int = 0,
        lines=(),
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.lines = list(lines)
        self.calls: list[dict] = []
        self.processes: list[_FakeProcess] = []

    async def create(self, *argv, **kwargs):
        call = {
            "argv": list(argv),
            "input": None,
            "cwd": kwargs.get("cwd"),
            "env": kwargs.get("env"),
        }
        self.calls.append(call)
        proc = _FakeProcess(self, call)
        self.processes.append(proc)
        return proc


@pytest.fixture
def claude(monkeypatch):
    """A ClaudeCLI whose async subprocess is captured instead of run."""
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    transport = _Transport()
    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec",
        transport.create,
    )
    return ClaudeCLI(), transport.calls, transport


def test_claude_cli_satisfies_the_protocol():
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    assert isinstance(ClaudeCLI(), LLM)


def test_factory_builds_it_under_both_names():
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    assert isinstance(build_llm("claude-cli"), ClaudeCLI)
    assert isinstance(build_llm("claude"), ClaudeCLI)


def test_every_call_suppresses_claude_md_memory_mcp_and_tools(claude):
    """Every call enforces the documented local-CLI isolation boundary."""
    llm, calls, _ = claude
    run(llm.complete(prompt="hi"))
    argv, env, cwd = calls[0]["argv"], calls[0]["env"], calls[0]["cwd"]

    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["CLAUDE_CODE_DISABLE_ORG_MEMORY"] == "1"
    assert "--strict-mcp-config" in argv
    assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert "--disallowed-tools" in argv
    for tool in ("Read", "Write", "Bash", "WebFetch", "Task"):
        assert tool in argv, f"{tool} was left enabled"
    assert cwd is not None and os.path.realpath(cwd) != os.path.realpath(os.getcwd())


def test_the_env_is_the_real_environment_plus_the_overrides(claude, monkeypatch):
    """PATH must survive, or the CLI's own node runtime is not findable."""
    monkeypatch.setenv("PATH", "/sentinel/bin")
    llm, calls, _ = claude
    run(llm.complete(prompt="hi"))
    assert calls[0]["env"]["PATH"] == "/sentinel/bin"


def test_complete_sends_the_prompt_on_stdin_and_returns_stripped_stdout(claude):
    llm, calls, transport = claude
    transport.stdout = "  the answer  \n"
    assert run(llm.complete(prompt="the question")) == "the answer"
    assert calls[0]["input"] == "the question"


def test_system_prompt_replaces_the_default(claude):
    llm, calls, _ = claude
    run(llm.complete(prompt="hi", system="You are a librarian."))
    argv = calls[0]["argv"]
    assert argv[argv.index("--system-prompt") + 1] == "You are a librarian."


def test_json_object_asks_for_json_without_losing_the_caller_system_prompt(claude):
    llm, calls, _ = claude
    run(llm.complete(
        prompt="hi", system="You are a librarian.", json_object=True
    ))
    sent = calls[0]["argv"][calls[0]["argv"].index("--system-prompt") + 1]
    assert sent.startswith("You are a librarian.")
    assert "JSON" in sent


def test_model_override_beats_the_constructor(claude):
    llm, calls, _ = claude
    run(llm.complete(prompt="hi", model="claude-opus-5"))
    assert calls[0]["argv"][calls[0]["argv"].index("--model") + 1] == "claude-opus-5"


def test_complete_json_constrains_the_cli_with_the_schema(claude):
    """The schema goes to the CLI, so the model is constrained rather than
    asked politely and then parsed hopefully."""
    llm, calls, transport = claude
    transport.stdout = '{"ok": true, "score": 0.75}'
    out = run(llm.complete_json(prompt="hi", schema=Verdict))
    assert out == Verdict(ok=True, score=0.75)

    import json as _json
    sent = _json.loads(calls[0]["argv"][calls[0]["argv"].index("--json-schema") + 1])
    assert sent == Verdict.model_json_schema()


def test_complete_json_raises_on_unparseable_output(claude):
    llm, _, transport = claude
    transport.stdout = "Sure! Here you go:"
    with pytest.raises(LLMError, match="unparseable JSON"):
        run(llm.complete_json(prompt="hi", schema=Verdict))


def test_complete_json_raises_when_the_shape_is_wrong(claude):
    """Valid JSON, wrong object -- must not reach the caller as a half-model."""
    llm, _, transport = claude
    transport.stdout = '{"ok": true, "score": 5.0}'
    with pytest.raises(LLMError, match="not a valid Verdict"):
        run(llm.complete_json(prompt="hi", schema=Verdict))


def test_a_nonzero_exit_becomes_llmerror_carrying_stderr(monkeypatch):
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    transport = _Transport(stderr="Not logged in", returncode=1)
    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec",
        transport.create,
    )
    with pytest.raises(LLMError, match="Not logged in"):
        run(ClaudeCLI().complete(prompt="hi"))


def test_a_missing_binary_says_how_to_fix_it(monkeypatch):
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    async def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec", boom
    )
    with pytest.raises(LLMError, match="not found. Install Claude Code and sign in"):
        run(ClaudeCLI().complete(prompt="hi"))


def test_a_timeout_reports_the_budget(monkeypatch):
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    entered = asyncio.Event()

    class HangingProcess(_FakeProcess):
        async def communicate(self, value: bytes) -> tuple[bytes, bytes]:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    transport = _Transport()

    async def create(*argv, **kwargs):
        call = {"argv": list(argv), "input": None, **kwargs}
        proc = HangingProcess(transport, call)
        transport.processes.append(proc)
        return proc

    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec", create
    )
    with pytest.raises(LLMError, match="timed out after"):
        run(ClaudeCLI(timeout_s=0.01).complete(prompt="hi"))
    assert transport.processes[0].terminated


def test_stream_yields_text_and_thinking_deltas_separately(monkeypatch):
    """The SSE layer sends token frames and reasoning frames down different
    channels, so the transport must keep them apart."""
    import json as _json

    from visionagent.providers.llm.claude_cli import ClaudeCLI

    def evt(kind: str, key: str, val: str) -> str:
        return _json.dumps({"type": "stream_event",
                            "event": {"type": "content_block_delta",
                                      "delta": {"type": kind, key: val}}})

    lines = [
        '{"type":"system","subtype":"init"}',          # ignored
        "not json at all",                             # ignored
        evt("thinking_delta", "thinking", "hmm"),
        evt("text_delta", "text", "one "),
        evt("text_delta", "text", "two"),
        '{"type":"result","is_error":false}',          # ignored
    ]

    transport = _Transport(lines=lines)
    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec",
        transport.create,
    )
    out = run(collect(ClaudeCLI().stream(prompt="hi")))
    assert out == [("", "hmm"), ("one ", ""), ("two", "")]


def test_stream_asks_the_cli_for_partial_messages(monkeypatch):
    """Without --include-partial-messages the CLI emits one final block and
    the UI would show nothing until the answer is complete."""
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    transport = _Transport()
    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec",
        transport.create,
    )
    run(collect(ClaudeCLI().stream(prompt="hi")))
    argv = transport.calls[0]["argv"]
    assert "--include-partial-messages" in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"


def test_cancelling_a_completion_terminates_and_reaps_the_cli(monkeypatch):
    """Cancellation must stop the real provider process, not just its waiter."""
    from visionagent.providers.llm.claude_cli import ClaudeCLI

    entered = asyncio.Event()
    transport = _Transport()

    class HangingProcess(_FakeProcess):
        async def communicate(self, value: bytes) -> tuple[bytes, bytes]:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    async def create(*argv, **kwargs):
        call = {"argv": list(argv), "input": None, **kwargs}
        proc = HangingProcess(transport, call)
        transport.processes.append(proc)
        return proc

    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec", create
    )

    async def cancel():
        task = asyncio.create_task(ClaudeCLI().complete(prompt="hi"))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(cancel())
    proc = transport.processes[0]
    assert proc.terminated and proc.returncode == -15


def test_closing_a_stream_terminates_and_reaps_the_cli(monkeypatch):
    """An HTTP client disconnect closes the provider's CLI child."""
    import json as _json

    from visionagent.providers.llm.claude_cli import ClaudeCLI

    line = _json.dumps({
        "type": "stream_event",
        "event": {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "one"},
        },
    })
    transport = _Transport(lines=[line])
    monkeypatch.setattr(
        "visionagent.providers.llm.claude_cli.asyncio.create_subprocess_exec",
        transport.create,
    )

    async def close():
        stream = ClaudeCLI().stream(prompt="hi")
        assert await anext(stream) == ("one", "")
        await stream.aclose()

    run(close())
    proc = transport.processes[0]
    assert proc.terminated and proc.returncode == -15


def test_a_foreign_model_name_falls_back_to_the_configured_one(claude):
    """A model name owned by another provider selects the configured default."""
    llm, calls, _ = claude
    run(llm.complete(prompt="hi", model="qwen-turbo"))
    argv = calls[0]["argv"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5"


def test_a_claude_model_name_is_still_honoured(claude):
    llm, calls, _ = claude
    for name in ("claude-opus-5", "sonnet", "haiku"):
        calls.clear()
        run(llm.complete(prompt="hi", model=name))
        argv = calls[0]["argv"]
        assert argv[argv.index("--model") + 1] == name, name


def test_the_foreign_name_is_warned_about_once(claude, caplog):
    import logging

    from visionagent.providers.llm.claude_cli import _FOREIGN_MODELS_SEEN

    _FOREIGN_MODELS_SEEN.discard("qwen-max")
    llm, _, _ = claude
    with caplog.at_level(logging.WARNING):
        run(llm.complete(prompt="hi", model="qwen-max"))
        run(llm.complete(prompt="hi", model="qwen-max"))
    assert sum("qwen-max" in r.getMessage() for r in caplog.records) == 1
