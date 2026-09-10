"""Local development/evaluation transport for the authenticated `claude` CLI.

The subprocess is isolated from project and user instructions, memory, MCP
servers, and tools. Authentication remains owned by the CLI application; this
adapter reads no API key. `scripts/check_claude_cli.py` verifies the isolation
contract after CLI upgrades. This process-based adapter is not a production
serving integration.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import tempfile
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from visionagent.providers.llm.base import LLMError

M = TypeVar("M", bound=BaseModel)

# Everything the CLI would otherwise offer the model. A text-completion backend
# must not be able to touch the filesystem or the network on its own initiative.
_TOOLS = (
    "Task Artifact Bash CronCreate CronDelete CronList DesignSync Edit "
    "EnterWorktree ExitWorktree Glob Grep Monitor NotebookEdit PushNotification "
    "Read RemoteTrigger ReportFindings ScheduleWakeup SendMessage Skill "
    "TaskOutput TaskStop TodoWrite ToolSearch WebFetch WebSearch Workflow Write"
).split()

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM = "You are a text-completion service. Answer only what is asked."

# Model identifiers are provider-specific. Foreign identifiers fall back to
# this adapter's configured model instead of failing a provider swap.
_FOREIGN_MODELS_SEEN: set[str] = set()

# These CLI-specific isolation controls are verified by check_claude_cli.py.
_ISOLATION_ENV = {
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_ORG_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


class ClaudeCLI:
    """Implements `visionagent.providers.llm.base.LLM`."""

    name = "claude-cli"

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-5",
        effort: str = "medium",
        binary: str = "claude",
        timeout_s: float = 300.0,
    ) -> None:
        self.model = model
        self.effort = effort
        self.binary = binary
        self.timeout_s = timeout_s

    def _resolve_model(self, model: str | None) -> str:
        """Resolve a provider-owned model name or use the configured default."""
        if not model:
            return self.model
        if model.startswith("claude") or model in ("opus", "sonnet", "haiku"):
            return model
        if model not in _FOREIGN_MODELS_SEEN:
            _FOREIGN_MODELS_SEEN.add(model)
            logger.warning(
                "%s does not own the model name %r; using %s. Model names are "
                "provider-specific and do not survive an LLM_PROVIDER swap.",
                self.name, model, self.model,
            )
        return self.model

    def _argv(self, *, system: str | None, model: str | None,
              extra: list[str] | None = None) -> list[str]:
        argv = [
            self.binary, "--print",
            "--model", self._resolve_model(model),
            "--effort", self.effort,
            "--system-prompt", system or _DEFAULT_SYSTEM,
            "--mcp-config", '{"mcpServers":{}}', "--strict-mcp-config",
            "--disallowed-tools", *_TOOLS,
        ]
        return argv + (extra or [])

    @staticmethod
    def _env() -> dict[str, str]:
        return {**os.environ, **_ISOLATION_ENV}

    async def _spawn(
        self,
        argv: list[str],
        *,
        cwd: str,
    ) -> asyncio.subprocess.Process:
        try:
            return await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=self._env(),
            )
        except FileNotFoundError as exc:
            raise LLMError(
                f"{self.name}: `{self.binary}` not found. Install Claude Code and sign in."
            ) from exc

    @staticmethod
    async def _terminate(proc: asyncio.subprocess.Process) -> None:
        """Stop and reap a child without leaving a zombie behind."""
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    async def _run(self, argv: list[str], prompt: str) -> str:
        # cwd is a fresh empty directory: the CLI discovers CLAUDE.md from the
        # working directory, and this project's would otherwise be prepended
        # to every completion.
        neutral = tempfile.mkdtemp(prefix="va-claude-")
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await self._spawn(argv, cwd=neutral)
            try:
                async with asyncio.timeout(self.timeout_s):
                    stdout, stderr = await proc.communicate(prompt.encode())
            except TimeoutError as exc:
                raise LLMError(
                    f"{self.name} timed out after {self.timeout_s:.0f}s"
                ) from exc
            if proc.returncode != 0:
                detail = stderr.decode(errors="replace").strip()[:300]
                raise LLMError(f"{self.name} exited {proc.returncode}: {detail}")
            return stdout.decode(errors="replace").strip()
        finally:
            # Cancellation reaches this block immediately. Unlike a blocking
            # subprocess in a worker thread, the real provider process is
            # terminated and reaped before the cancelled call exits.
            if proc is not None:
                await self._terminate(proc)
            shutil.rmtree(neutral, ignore_errors=True)

    # ---------------------------------------------------------------- protocol
    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        json_object: bool = False,
    ) -> str:
        # `temperature` has no CLI equivalent and is accepted for Protocol
        # compatibility rather than silently pretended to.
        sys_prompt = system or _DEFAULT_SYSTEM
        if json_object:
            sys_prompt += " Respond with a single JSON object and nothing else."
        return await self._run(self._argv(system=sys_prompt, model=model), prompt)

    async def stream(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[tuple[str, str]]:
        """Real token deltas, via the CLI's stream-json output."""
        argv = self._argv(
            system=system, model=model,
            extra=["--output-format", "stream-json", "--include-partial-messages", "--verbose"],
        )
        neutral = tempfile.mkdtemp(prefix="va-claude-")
        proc: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[bytes] | None = None
        try:
            proc = await self._spawn(argv, cwd=neutral)
            assert proc.stdin is not None
            assert proc.stdout is not None
            assert proc.stderr is not None

            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            # Drain stderr concurrently. Waiting until stdout closes can
            # deadlock when a verbose child fills the stderr pipe.
            stderr_task = asyncio.create_task(proc.stderr.read())
            failure = ""
            try:
                async with asyncio.timeout(self.timeout_s):
                    while line_bytes := await proc.stdout.readline():
                        line = line_bytes.decode(errors="replace").strip()
                        if not line.startswith("{"):
                            continue
                        try:
                            frame = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if frame.get("type") == "result" and frame.get("is_error"):
                            failure = str(
                                frame.get("result") or frame.get("subtype") or ""
                            )[:300]
                            continue
                        if frame.get("type") != "stream_event":
                            continue
                        event = frame.get("event", {})
                        if event.get("type") != "content_block_delta":
                            continue
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield (delta.get("text", ""), "")
                        elif delta.get("type") == "thinking_delta":
                            yield ("", delta.get("thinking", ""))
                    await proc.wait()
            except TimeoutError as exc:
                raise LLMError(
                    f"{self.name} timed out after {self.timeout_s:.0f}s"
                ) from exc

            stderr = (await stderr_task).decode(errors="replace").strip()[:300]
            if proc.returncode != 0:
                raise LLMError(
                    f"{self.name} stream exited {proc.returncode}: {failure or stderr}"
                )
        finally:
            # Runs on normal exhaustion, aclose(), timeout and task
            # cancellation. The child never survives its request.
            if proc is not None:
                await self._terminate(proc)
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task
            shutil.rmtree(neutral, ignore_errors=True)

    async def complete_json(
        self,
        *,
        prompt: str,
        schema: type[M],
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> M:
        """Uses the CLI's own `--json-schema`, so the model is constrained
        rather than asked nicely and then parsed."""
        argv = self._argv(
            system=system, model=model,
            extra=["--json-schema", json.dumps(schema.model_json_schema())],
        )
        raw = await self._run(argv, prompt)
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{self.name} returned unparseable JSON for {schema.__name__}: {raw[:200]!r}"
            ) from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMError(
                f"{self.name} returned JSON that is not a valid {schema.__name__}: {exc}"
            ) from exc

    async def aclose(self) -> None:
        """No persistent transport is retained between CLI invocations."""
