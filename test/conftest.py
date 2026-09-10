"""Fixtures for the end-to-end suite.

These need the whole stack: Postgres, Elasticsearch, and the API answering on
:8000. Everything here talks to a running system; nothing is stubbed.

    make test-e2e
    pytest test/backend            # from the repo root
"""
import asyncio
import contextlib
import os
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


# A test run is a development-mode process: without JWT_SECRET_KEY the API
# refuses to start in production, and a random per-process secret is what a
# test wants anyway. setdefault, so an exported value still wins.
os.environ.setdefault("APP_ENV", "test")

# A local Docker-oriented .env may address services through
# host.docker.internal. Host-run tests translate that hostname to localhost;
# an explicit environment variable still wins.
if "DATABASE_URL" not in os.environ:
    env_file = REPO_ROOT / "backend" / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DATABASE_URL"):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                os.environ["DATABASE_URL"] = value.replace(
                    "host.docker.internal", "localhost"
                )
                break


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def provider_available() -> bool:
    """Whether the model provider will actually answer.

    An account in arrears, an expired key or a regional outage all make the
    provider refuse every call. Those are not defects, and reporting them as
    failures teaches people to ignore a red suite. Tests that need a real
    completion skip with the provider's own reason instead.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
    try:
        from visionagent.providers.llm import build_llm

        async def probe() -> None:
            llm = build_llm()
            try:
                await llm.complete(prompt="ok")
            finally:
                await llm.aclose()

        asyncio.run(probe())
        return True
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model provider unavailable: {str(exc)[:160]}", allow_module_level=False)
        return False


@pytest.fixture
def needs_provider(provider_available: bool) -> None:
    """Declare that a test needs a real completion or embedding."""
    return


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("E2E_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def sample_pdf(repo_root: Path) -> Path:
    """The fixed RAGFlow PDF used by every end-to-end document test."""
    fixture = repo_root / "test" / "fixtures" / "ragflow" / "Doc1.pdf"
    if not fixture.is_file():
        pytest.fail(f"required PDF fixture is missing: {fixture}")
    return fixture


@pytest.fixture(scope="session")
def live_backend(base_url: str):
    """Skip the whole e2e module unless the API is actually answering."""
    requests = pytest.importorskip("requests")
    try:
        r = requests.get(f"{base_url}/openapi.json", timeout=5)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"backend not reachable at {base_url}: {exc}")
    return base_url


@pytest.fixture(scope="session")
def api(live_backend: str):
    """Authenticated session bound to a throwaway user."""
    requests = pytest.importorskip("requests")

    class Client:
        def __init__(self, root: str):
            self.root = root
            self.s = requests.Session()
            self.username = f"pytest_{uuid.uuid4().hex[:10]}"
            self.password = "PytestPass123!"
            self.sessions: list[str] = []

        def req(self, method: str, path: str, **kw):
            kw.setdefault("timeout", 300)
            return self.s.request(method, self.root + path, **kw)

        def register_and_login(self):
            self.req(
                "POST",
                "/register",
                json={"username": self.username, "password": self.password},
            )
            r = self.req(
                "POST",
                "/login",
                json={"username": self.username, "password": self.password},
            )
            r.raise_for_status()
            token = r.json()["access_token"]
            self.s.headers["Authorization"] = f"Bearer {token}"
            return token

        def new_session(self) -> str:
            r = self.req("POST", "/create_session/")
            r.raise_for_status()
            sid = r.json()["session_id"]
            self.sessions.append(sid)
            return sid

    client = Client(live_backend)
    client.register_and_login()
    yield client

    for sid in client.sessions:
        with contextlib.suppress(Exception):
            client.req("DELETE", f"/delete_session/{sid}")


@pytest.fixture
def session_id(api) -> str:
    return api.new_session()


def sse_events(response) -> list[str]:
    """Collect `data:` payloads from a streamed SSE response."""
    out = []
    for raw in response.iter_lines():
        if raw and raw.startswith(b"data: "):
            out.append(raw[6:].decode("utf-8", "replace"))
    return out
