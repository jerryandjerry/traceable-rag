#!/usr/bin/env python
"""Maintain Traceable RAG's offline compatibility baseline.

Usage from the repository root::

    make golden          # deliberately replace the reviewed baseline
    make golden-check    # read-only replay and exact comparison

The baseline covers deterministic local contracts only: DeepDoc parsing, text
analysis, and the Elasticsearch document shape. It never loads application
settings, connects to a datastore, or calls a hosted provider. Live answer
quality is measured separately by ``make test-answer-quality``.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parents[1] / "tests" / "golden" / "baseline.json"
SCHEMA_VERSION = 3
LEGACY_SCHEMA_VERSION = 2
PERSISTED_VECTOR_DIM = 1024
SNAPSHOT_PREFIX = "VISIONAGENT_GOLDEN_SNAPSHOT="
PLATFORM_KEY_PATTERN = re.compile(r"^[a-z0-9_]+-[a-z0-9_]+-py[1-9][0-9]*$")
IGNORED_APPLICATION_ENV = (
    "APP_ENV",
    "JWT_SECRET_KEY",
    "DATABASE_URL",
    "DASHSCOPE_API_KEY",
    "SERPER_API_KEY",
    "PARSER",
    "CHUNKSTORE",
    "EMBEDDING_PROVIDER",
    "RAG_PROJECT_BASE",
    "RAG_DEPLOY_BASE",
)

# Fixed, versioned Apache-2.0 fixtures from RAGFlow's benchmark suite.
PARSER_FIXTURE_DIR = "test/fixtures/ragflow"
PARSER_FIXTURES = (
    "Doc1.pdf",
    "Doc2.pdf",
    "Doc3.pdf",
)

RUNTIME_ASSETS = (
    "backend/src/visionagent/vendor/ragflow/9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
    "backend/src/visionagent/vendor/ragflow/rag/res/deepdoc/det.onnx",
    "backend/src/visionagent/vendor/ragflow/rag/res/deepdoc/layout.onnx",
    "backend/src/visionagent/vendor/ragflow/rag/res/deepdoc/ocr.res",
    "backend/src/visionagent/vendor/ragflow/rag/res/deepdoc/rec.onnx",
    "backend/src/visionagent/vendor/ragflow/rag/res/deepdoc/tsr.onnx",
    "backend/src/visionagent/vendor/ragflow/rag/res/deepdoc/updown_concat_xgb.model",
    "backend/src/visionagent/vendor/ragflow/rag/res/huqie.txt",
    "backend/src/visionagent/vendor/ragflow/rag/res/huqie.txt.trie",
)

# Fixed corpus for the analyzer. Deliberately includes plural/stem collapsing,
# a measurement, a zoning code, and Chinese segmentation.
ANALYZER_CORPUS = (
    "The sidewalks widths shall be 1.8m minimum",
    "Curb extensions reduce crossing distances",
    "Accessibility studies analysing pedestrian R-4 zoning",
    "Tangent curb taper",
    "建筑设计规范",
    "2+2",
    "",
)


class GoldenError(RuntimeError):
    """The compatibility baseline cannot be produced or trusted."""


def _platform_part(value: str, *, label: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower()).strip("_")
    if not normalized:
        raise GoldenError(f"cannot determine golden {label}")
    return normalized


def parser_platform_key(
    *,
    system_name: str | None = None,
    machine_name: str | None = None,
    python_version: tuple[int, int] | None = None,
) -> str:
    """Return the normalized native-runtime key for exact parser snapshots."""
    system_part = _platform_part(system_name or platform.system(), label="operating system")
    machine_part = _platform_part(machine_name or platform.machine(), label="architecture")
    machine_part = {
        "aarch64": "arm64",
        "amd64": "x86_64",
        "x64": "x86_64",
    }.get(machine_part, machine_part)
    major, minor = python_version or (sys.version_info.major, sys.version_info.minor)
    if major < 1 or minor < 0:
        raise GoldenError(f"invalid Python version for golden platform key: {major}.{minor}")
    return f"{system_part}-{machine_part}-py{major}{minor}"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_file(relative: str) -> Path:
    path = REPO / relative
    if not path.is_file() or path.stat().st_size == 0:
        raise GoldenError(f"required golden input is missing or empty: {relative}")
    return path


def _parser_fixture(name: str) -> Path:
    return _required_file(f"{PARSER_FIXTURE_DIR}/{name}")


def record_analyzer() -> dict[str, dict[str, str]]:
    from visionagent.vendor.ragflow.rag.nlp import rag_tokenizer

    out: dict[str, dict[str, str]] = {}
    for text in ANALYZER_CORPUS:
        coarse = rag_tokenizer.tokenize(text)
        out[text] = {
            "content_tokens": coarse,
            "content_tokens_fine": rag_tokenizer.fine_grained_tokenize(coarse),
        }
    return out


def _document_snapshot(path: Path) -> dict[str, Any]:
    """Parse one fixture in the current worker and return canonical JSON data."""
    from visionagent.service.parsers.deepdoc import DeepDocParser

    chunks = DeepDocParser().parse(file_path=path, file_name=path.name)
    if not chunks:
        raise GoldenError(f"golden parser produced no chunks: {path.name}")

    return {
        "source_sha256": sha_file(path),
        "chunk_count": len(chunks),
        "chunks": [
            {
                "content_sha256": sha_text(chunk.content),
                "content_tokens": chunk.content_tokens,
                "content_tokens_fine": chunk.content_tokens_fine,
                "page_nums": chunk.page_nums,
                "top_offsets": chunk.top_offsets,
                "doc_name_tokens": chunk.doc_name_tokens,
                "has_image": bool(chunk.image),
                "ref_image_count": len(chunk.ref_images),
            }
            for chunk in chunks
        ],
    }


def _worker_environment() -> dict[str, str]:
    """A stable local worker environment with application selectors removed."""
    env = os.environ.copy()
    for name in IGNORED_APPLICATION_ENV:
        env.pop(name, None)
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _snapshot_in_clean_process(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--snapshot", path.name],
        cwd=REPO,
        env=_worker_environment(),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if result.returncode != 0:
        detail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-30:])
        raise GoldenError(f"parser worker failed for {path.name}:\n{detail}")

    payloads = [
        line.removeprefix(SNAPSHOT_PREFIX)
        for line in result.stdout.splitlines()
        if line.startswith(SNAPSHOT_PREFIX)
    ]
    if len(payloads) != 1:
        raise GoldenError(
            f"parser worker returned {len(payloads)} snapshots for {path.name}; expected one"
        )
    try:
        snapshot = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise GoldenError(f"parser worker returned malformed JSON for {path.name}") from exc
    if not isinstance(snapshot, dict):
        raise GoldenError(f"parser worker returned a non-object for {path.name}")
    return snapshot


def record_parse() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in PARSER_FIXTURES:
        path = _parser_fixture(name)
        snapshot = _snapshot_in_clean_process(path)
        out[name] = snapshot
        print(f"  {name}: {snapshot['chunk_count']} chunks")
    return out


def record_assets() -> dict[str, str]:
    return {relative: sha_file(_required_file(relative)) for relative in RUNTIME_ASSETS}


def record_es_document_shape() -> dict[str, Any]:
    """Shape one document with local fakes; no database or provider is loaded."""
    from visionagent.models import ParsedChunk
    from visionagent.service.vectorstore.elasticsearch.store import ElasticsearchChunkStore

    class FixedAnalyzer:
        name = "golden-fixed"

        def analyze(self, text: str) -> str:
            raise AssertionError("golden shape unexpectedly invoked the analyzer")

        def analyze_fine(self, text: str) -> str:
            raise AssertionError("golden shape unexpectedly invoked the analyzer")

        def synonyms(self, term: str) -> list[str]:
            raise AssertionError("golden shape unexpectedly invoked the analyzer")

    class ZeroEmbedder:
        name = "golden-zero"
        dimensions = PERSISTED_VECTOR_DIM

        def embed(self, text: str) -> list[float]:
            return [0.0] * self.dimensions

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed(text) for text in texts]

        async def aembed(self, text: str) -> list[float]:
            return self.embed(text)

        async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
            return self.embed_batch(texts)

        async def aclose(self) -> None:
            return None

    class RejectingWriter:
        def index(self, *, index_name: str, documents: list[dict[str, Any]]) -> int:
            raise AssertionError("golden shape attempted a database write")

        def delete_document(self, *, index_name: str, doc_name: str) -> int:
            raise AssertionError("golden shape attempted a database delete")

        def delete_index(self, *, index_name: str) -> int:
            raise AssertionError("golden shape attempted a database delete")

    store = ElasticsearchChunkStore(
        analyzer=FixedAnalyzer(),
        embedder=ZeroEmbedder(),
        store=RejectingWriter(),
    )
    chunk = ParsedChunk(
        id="goldenchunk0001",
        content_tokens="sidewalk width",
        content="Sidewalk widths.",
        content_tokens_fine="sidewalk width",
        page_nums=[1],
        top_offsets=[10],
        doc_name_tokens="golden doc",
    )
    doc = store.to_document(chunk, index_name="golden-user", doc_name="golden.pdf")
    vector_fields = sorted(key for key in doc if key.endswith("_vec"))
    return {
        "fields": sorted(key for key in doc if not key.endswith("_vec")),
        "vector_fields": vector_fields,
        "vector_dim": len(doc[vector_fields[0]]) if vector_fields else 0,
    }


def _record_platform_contract(platform_key: str) -> dict[str, Any]:
    print("recording analyzer baseline...")
    analyzer = record_analyzer()
    print(f"recording parser baseline for {platform_key}...")
    parsed = record_parse()
    print("recording parser and tokenizer asset identities...")
    assets = record_assets()
    print("recording Elasticsearch document shape...")
    es_shape = record_es_document_shape()
    return {
        "schema_version": SCHEMA_VERSION,
        "analyzer": analyzer,
        "parse_by_platform": {platform_key: parsed},
        "runtime_assets": assets,
        "es_document": es_shape,
    }


def _validate_parser_snapshots(platform_key: str, snapshots: Any) -> None:
    if not isinstance(snapshots, dict) or set(snapshots) != set(PARSER_FIXTURES):
        raise GoldenError(
            f"golden parser fixture set for {platform_key} does not match the fixed manifest"
        )
    for name, snapshot in snapshots.items():
        if not isinstance(snapshot, dict):
            raise GoldenError(f"golden parser snapshot is not an object: {platform_key}/{name}")
        chunks = snapshot.get("chunks")
        if not isinstance(chunks, list) or snapshot.get("chunk_count") != len(chunks):
            raise GoldenError(f"golden chunk count is inconsistent for {platform_key}/{name}")
        source_hash = snapshot.get("source_sha256", "")
        if not isinstance(source_hash, str) or len(source_hash) != 64:
            raise GoldenError(f"golden source hash is invalid for {platform_key}/{name}")
        for index, chunk in enumerate(chunks):
            content_hash = chunk.get("content_sha256", "") if isinstance(chunk, dict) else ""
            if not isinstance(content_hash, str) or len(content_hash) != 64:
                raise GoldenError(
                    f"golden content hash is invalid for {platform_key}/{name} chunk {index}"
                )


def _validated_platform_snapshots(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise GoldenError("golden baseline contains no platform parser snapshots")
    for platform_key, snapshots in value.items():
        if not isinstance(platform_key, str) or not PLATFORM_KEY_PATTERN.fullmatch(platform_key):
            raise GoldenError(f"invalid golden parser platform key: {platform_key!r}")
        machine_part = platform_key.split("-", 1)[1].rsplit("-py", 1)[0]
        if machine_part in {"aarch64", "amd64", "x64"}:
            raise GoldenError(f"non-normalized golden parser platform key: {platform_key!r}")
        _validate_parser_snapshots(platform_key, snapshots)
    return value


def _validate_baseline(baseline: Any) -> dict[str, Any]:
    if not isinstance(baseline, dict):
        raise GoldenError("golden baseline root must be an object")
    expected_keys = {
        "schema_version",
        "analyzer",
        "parse_by_platform",
        "runtime_assets",
        "es_document",
    }
    if set(baseline) != expected_keys:
        raise GoldenError("golden baseline fields do not match schema 3")
    if baseline.get("schema_version") != SCHEMA_VERSION:
        raise GoldenError(
            f"unsupported golden schema {baseline.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if set(baseline.get("analyzer", {})) != set(ANALYZER_CORPUS):
        raise GoldenError("golden analyzer corpus is incomplete or contains unknown cases")
    _validated_platform_snapshots(baseline.get("parse_by_platform"))
    if set(baseline.get("runtime_assets", {})) != set(RUNTIME_ASSETS):
        raise GoldenError("golden runtime asset manifest is incomplete")
    es_shape = baseline.get("es_document", {})
    if es_shape.get("vector_dim") != PERSISTED_VECTOR_DIM:
        raise GoldenError("golden vector dimension does not match the persisted contract")
    return baseline


def _canonical(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def _selected_platform_contract(
    baseline: dict[str, Any], platform_key: str
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analyzer": baseline["analyzer"],
        "parse_by_platform": {
            platform_key: baseline["parse_by_platform"][platform_key]
        },
        "runtime_assets": baseline["runtime_assets"],
        "es_document": baseline["es_document"],
    }


def _load_existing_parser_snapshots_for_update() -> dict[str, Any]:
    if not BASELINE.is_file():
        return {}
    try:
        loaded = json.loads(BASELINE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GoldenError("golden baseline is not valid JSON") from exc

    schema_version = loaded.get("schema_version") if isinstance(loaded, dict) else None
    if schema_version == SCHEMA_VERSION:
        # An update deliberately replaces shared contracts, so only validate
        # the platform snapshots that will be preserved. `--check` remains
        # strict over every schema field.
        return dict(_validated_platform_snapshots(loaded.get("parse_by_platform")))
    if schema_version == LEGACY_SCHEMA_VERSION and isinstance(loaded, dict):
        # Schema 2 had one unlabelled native snapshot. Validate it as the
        # current platform, then replace it with a fresh labelled recording.
        _validate_parser_snapshots(parser_platform_key(), loaded.get("parse"))
        return {}
    raise GoldenError(
        f"unsupported golden schema {schema_version!r}; expected {SCHEMA_VERSION}"
    )


def check() -> int:
    if not BASELINE.is_file():
        raise GoldenError(f"golden baseline is missing: {BASELINE.relative_to(REPO)}")
    try:
        expected = _validate_baseline(json.loads(BASELINE.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise GoldenError("golden baseline is not valid JSON") from exc

    platform_key = parser_platform_key()
    if platform_key not in expected["parse_by_platform"]:
        raise GoldenError(
            f"golden parser snapshot is missing for {platform_key}; "
            "run make golden on that platform and review the new snapshot"
        )
    selected_expected = _selected_platform_contract(expected, platform_key)
    actual = _validate_baseline(_record_platform_contract(platform_key))
    if actual != selected_expected:
        print("GOLDEN DRIFT:")
        diff = difflib.unified_diff(
            _canonical(selected_expected).splitlines(),
            _canonical(actual).splitlines(),
            fromfile="recorded baseline",
            tofile="current output",
            lineterm="",
        )
        for line in diff:
            print(line)
        return 1

    snapshots = actual["parse_by_platform"][platform_key]
    chunks = sum(item["chunk_count"] for item in snapshots.values())
    print(
        f"golden baseline reproduces for {platform_key}: "
        f"{len(snapshots)}/{len(PARSER_FIXTURES)} documents, {chunks} chunks, "
        f"{len(actual['analyzer'])} analyzer cases"
    )
    return 0


def update() -> int:
    preserved = _load_existing_parser_snapshots_for_update()
    platform_key = parser_platform_key()
    recorded = _record_platform_contract(platform_key)
    current_snapshots = recorded["parse_by_platform"][platform_key]
    recorded["parse_by_platform"] = {**preserved, platform_key: current_snapshots}
    candidate = _validate_baseline(recorded)
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=BASELINE.parent,
        prefix=".baseline.",
        suffix=".json",
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_canonical(candidate))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, BASELINE)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass

    snapshots = candidate["parse_by_platform"][platform_key]
    chunks = sum(item["chunk_count"] for item in snapshots.values())
    print(f"\nwrote {BASELINE.relative_to(REPO)} atomically")
    print(f"  analyzer : {len(candidate['analyzer'])} inputs")
    print(
        f"  parse    : {platform_key}: {len(snapshots)} documents, {chunks} chunks "
        f"({len(candidate['parse_by_platform']) - 1} other platform snapshots preserved)"
    )
    print(f"  es fields: {len(candidate['es_document']['fields'])}")
    return 0


def main() -> int:
    try:
        # The baseline is deliberately hermetic. Selectors belonging to the
        # running application cannot alter either the parent or worker path.
        for name in IGNORED_APPLICATION_ENV:
            os.environ.pop(name, None)
        if len(sys.argv) == 3 and sys.argv[1] == "--snapshot":
            name = sys.argv[2]
            if name not in PARSER_FIXTURES:
                raise GoldenError(f"unknown golden fixture: {name}")
            snapshot = _document_snapshot(_parser_fixture(name))
            print(SNAPSHOT_PREFIX + json.dumps(snapshot, ensure_ascii=False, sort_keys=True))
            return 0
        if sys.argv[1:] == ["--check"]:
            return check()
        if sys.argv[1:]:
            raise GoldenError(f"unknown arguments: {' '.join(sys.argv[1:])}")
        return update()
    except (GoldenError, subprocess.TimeoutExpired) as exc:
        print(f"golden error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
