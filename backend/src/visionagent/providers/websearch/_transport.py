"""Small HTTP transport helpers shared by web-search providers."""
from __future__ import annotations

from typing import Any

import httpx


async def read_response(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> bytes:
    """Read one response and always release it, including on cancellation."""
    request = client.build_request(method, url, **kwargs)
    response = await client.send(request, stream=True)
    try:
        response.raise_for_status()
        return await response.aread()
    finally:
        await response.aclose()
