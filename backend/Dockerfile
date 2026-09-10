FROM python:3.11.16-slim-bookworm@sha256:528257d48c1da0dcecc2e725d1ae34498d60c965f1241e39cd6a85a8859bdf84

# Keep the installer version aligned with the lockfile toolchain.
COPY --from=ghcr.io/astral-sh/uv:0.10.2@sha256:94a23af2d50e97b87b522d3cea24aaf8a1faedec1344c952767434f69585cbf9 /uv /uvx /bin/

# Native build and document-rendering dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    make \
    libgl1 \
    libgl1-mesa-glx \
    libglu1-mesa \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/srv/.venv/bin:$PATH"

# Resolve nothing in the image: install the exact production dependency set
# recorded in uv.lock. Source changes then reuse this dependency layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

COPY src ./src
COPY scripts ./scripts
RUN uv sync --locked --no-dev --no-editable --no-cache

# NLTK corpora (punkt, wordnet) are required by the text analyzer and are not
# pip dependencies.
RUN python scripts/setup_nltk.py

EXPOSE 8000
CMD ["uvicorn", "visionagent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
