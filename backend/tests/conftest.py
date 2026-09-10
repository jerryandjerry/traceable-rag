"""Fixtures for the backend unit tests.

The live-stack fixtures (base_url, live_backend, api, session_id,
sample_pdf) live in test/conftest.py, beside the e2e suite that needs
them -- a unit test that reaches a live service is not a unit test.
"""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/tests -> backend -> repo
BACKEND_DIR = REPO_ROOT / "backend"
APP_DIR = BACKEND_DIR / "src" / "visionagent"

# No sys.path manipulation: the backend is an installed package
# (`cd backend && uv sync --extra dev`), so imports resolve from any CWD.

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
def app_dir() -> Path:
    """The importable package root: backend/src/visionagent."""
    return APP_DIR


@pytest.fixture(scope="session")
def backend_dir() -> Path:
    """The Python project root containing pyproject.toml."""
    return BACKEND_DIR
