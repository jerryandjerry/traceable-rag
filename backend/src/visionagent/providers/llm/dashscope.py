"""DashScope via its OpenAI-compatible endpoint.

The adapter preserves the configured base URL, model selection, and streaming
shape behind the shared LLM contract.
"""
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from visionagent.config.settings import settings
from visionagent.providers.llm.base import LLMError

M = TypeVar("M", bound=BaseModel)

# Model responses may wrap JSON in prose or fenced blocks.
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_SPAN = re.compile(r"[\[{].*[\]}]", re.DOTALL)


def extract_json(raw: str) -> str:
    """Pull the JSON payload out of a model response."""
    if not raw:
        raise LLMError("empty response")
    fenced = _FENCE.search(raw)
    if fenced:
        return fenced.group(1)
    span = _SPAN.search(raw)
    if span:
        return span.group(0)
    return raw.strip()


class DashScopeLLM:
    """Implements `visionagent.providers.llm.base.LLM`."""

    name = "dashscope"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> None:
        self._default_model = default_model or settings.chat_model
        self._client = AsyncOpenAI(
            api_key=api_key or settings.dashscope_api_key,
            base_url=base_url or settings.dashscope_base_url,
        )

    def _messages(self, prompt: str, system: str | None) -> Any:
        msgs = [{"role": "system", "content": system}] if system else []
        msgs.append({"role": "user", "content": prompt})
        return msgs

    async def complete(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        json_object: bool = False,
    ) -> str:
        # The SDK overloads create() on the literal value of `stream`, so a
        # dict of extra kwargs defeats overload resolution. Typed loosely here
        # rather than reproducing the SDK's TypedDict unions.
        extra: dict[str, Any] = (
            {"response_format": {"type": "json_object"}} if json_object else {}
        )
        try:
            create: Any = self._client.chat.completions.create
            completion = await create(
                model=model or self._default_model,
                messages=self._messages(prompt, system),
                temperature=temperature,
                stream=False,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001 -- provider errors are opaque
            raise LLMError(f"{self.name} completion failed: {exc}") from exc
        return completion.choices[0].message.content or ""

    async def stream(
        self,
        *,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> AsyncIterator[tuple[str, str]]:
        try:
            completion: Any = await self._client.chat.completions.create(
                model=model or self._default_model,
                messages=self._messages(prompt, system),
                temperature=temperature,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"{self.name} stream failed: {exc}") from exc

        # AsyncStream.__aexit__ closes the HTTP response on normal exhaustion,
        # generator close, timeout, and task cancellation.
        async with completion:
            async for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                yield (
                    getattr(delta, "content", None) or "",
                    getattr(delta, "reasoning_content", None) or "",
                )

    async def complete_json(
        self,
        *,
        prompt: str,
        schema: type[M],
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> M:
        raw = await self.complete(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            json_object=True,
        )
        payload = extract_json(raw)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"{self.name} returned unparseable JSON for {schema.__name__}: {payload[:200]!r}"
            ) from exc
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMError(
                f"{self.name} returned JSON that is not a valid {schema.__name__}: {exc}"
            ) from exc

    async def aclose(self) -> None:
        """Close the shared HTTP transport."""
        await self._client.close()
