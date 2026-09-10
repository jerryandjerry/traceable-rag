"""EmbeddingGemma-300m, run locally.

No API key and no network: the weights sit in `models/embeddinggemma-300m` and
sentence-transformers runs them in this process.

It is not a drop-in swap for `online`. The vectors are **768 wide** where
DashScope's `text-embedding-v4` is configured for 1024, and Elasticsearch keys
the field off the width (`q_768_vec` vs `q_1024_vec`), so documents indexed
with one provider are invisible to queries made with the other. Switching means
re-indexing.
"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

from visionagent.providers.embedding.base import EmbeddingError

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "embeddinggemma-300m"

# One loaded model per (path, device), for the life of the process.
_MODELS: dict[tuple[str, str | None], Any] = {}
_LOCK = threading.Lock()


def _load(path: str, device: str | None) -> Any:
    """Return one process-cached SentenceTransformer per path and device."""
    key = (path, device)
    with _LOCK:
        if key not in _MODELS:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "the offline embedder needs sentence-transformers and torch: "
                    "uv sync --project backend --locked --extra offline"
                ) from exc
            if not Path(path).is_dir():
                raise EmbeddingError(f"no embedding model at {path}")

            # DeepDoc and torch load separate OpenMP runtimes in one worker;
            # limiting torch avoids contention between their CPU thread pools.
            import torch
            torch.set_num_threads(1)

            _MODELS[key] = SentenceTransformer(path, device=device)
        return _MODELS[key]


class OfflineEmbedder:
    """Implements `visionagent.providers.embedding.base.Embedder`."""

    name = "offline"

    # No unconditional task prefix: the interface does not distinguish query
    # from passage roles, and compatible local models may define no prompts.

    # sentence-transformers batches internally; this bounds peak memory.
    BATCH_SIZE = 16

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        device: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        self.model_path = str(model_path or os.getenv("EMBEDDING_MODEL_PATH") or _DEFAULT_MODEL_DIR)
        self.device = device or os.getenv("EMBEDDING_DEVICE") or None
        self._model = _load(self.model_path, self.device)

        # Support both dimension accessors in the declared version range.
        get_dim = getattr(self._model, "get_embedding_dimension", None) \
            or self._model.get_sentence_embedding_dimension
        width = int(get_dim())

        # Reported rather than configured. The width is a property of the
        # weights, and store.py names the Elasticsearch field after the vector
        # it is handed, so accepting a different number here would quietly
        # write to a field nothing queries.
        if dimensions is not None and dimensions != width:
            raise EmbeddingError(
                f"{self.model_path} produces {width}-wide vectors, not {dimensions}"
            )
        self.dimensions = width

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            vectors = self._model.encode(
                texts,
                batch_size=self.BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # noqa: BLE001 -- torch errors are opaque
            raise EmbeddingError(f"{self.name} embedding failed: {exc}") from exc

        # The model ends in a Normalize module, so these are already unit
        # length; float() because Elasticsearch is handed JSON, not numpy.
        return [[float(x) for x in vector] for vector in vectors]

    async def aembed(self, text: str) -> list[float]:
        # sentence-transformers is CPU/GPU synchronous. This is the explicit
        # local-compute bridge; unlike hosted I/O, it has no async transport.
        return await asyncio.to_thread(self.embed, text)

    async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_batch, texts)

    async def aclose(self) -> None:
        return None
