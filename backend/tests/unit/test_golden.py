"""Replay the recorded offline behaviour baseline and diff.

The rest of the suite asserts that things work. This asserts that the parser,
analyzer and persisted document contract remain explicitly compatible. Update
the reviewed baseline with:

    make golden        # or: python backend/scripts/record_golden.py

A deliberate behaviour change requires a reviewed baseline update in the same
change; unexplained drift is a failure.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

BASELINE_PATH = Path(__file__).resolve().parents[1] / "golden" / "baseline.json"
SCHEMA_VERSION = 3
SNAPSHOT_PREFIX = "VISIONAGENT_GOLDEN_SNAPSHOT="


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE_PATH.is_file(), "golden baseline missing; run make golden"
    loaded = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert loaded.get("schema_version") == SCHEMA_VERSION
    return loaded


@pytest.fixture(scope="module")
def golden_script(repo_root: Path) -> ModuleType:
    path = repo_root / "backend" / "scripts" / "record_golden.py"
    spec = importlib.util.spec_from_file_location("golden_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parser_platform_key_normalizes_runtime_aliases(golden_script: ModuleType):
    assert golden_script.parser_platform_key(
        system_name="Darwin",
        machine_name="aarch64",
        python_version=(3, 11),
    ) == "darwin-arm64-py311"
    assert golden_script.parser_platform_key(
        system_name="Linux",
        machine_name="AMD64",
        python_version=(3, 12),
    ) == "linux-x86_64-py312"


def test_baseline_rejects_non_normalized_platform_keys(
    golden_script: ModuleType, baseline: dict
):
    invalid = json.loads(json.dumps(baseline))
    snapshots = next(iter(invalid["parse_by_platform"].values()))
    invalid["parse_by_platform"] = {"linux-amd64-py311": snapshots}

    with pytest.raises(golden_script.GoldenError, match="non-normalized"):
        golden_script._validate_baseline(invalid)


def test_update_preserves_other_platform_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden_script: ModuleType,
    baseline: dict,
):
    current_key = "darwin-arm64-py311"
    other_key = "linux-x86_64-py311"
    existing = json.loads(json.dumps(baseline))
    current_snapshots = next(iter(existing["parse_by_platform"].values()))
    preserved_snapshots = json.loads(json.dumps(current_snapshots))
    existing["parse_by_platform"] = {
        current_key: current_snapshots,
        other_key: preserved_snapshots,
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(existing), encoding="utf-8")

    replacement = json.loads(json.dumps(current_snapshots))
    replacement["Doc1.pdf"]["chunks"][0]["content_sha256"] = "f" * 64
    recorded = golden_script._selected_platform_contract(existing, current_key)
    recorded["parse_by_platform"][current_key] = replacement

    monkeypatch.setattr(golden_script, "BASELINE", baseline_path)
    monkeypatch.setattr(golden_script, "REPO", tmp_path)
    monkeypatch.setattr(golden_script, "parser_platform_key", lambda: current_key)
    monkeypatch.setattr(
        golden_script,
        "_record_platform_contract",
        lambda platform_key: recorded,
    )

    assert golden_script.update() == 0
    updated = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert updated["parse_by_platform"][current_key] == replacement
    assert updated["parse_by_platform"][other_key] == preserved_snapshots


def test_check_compares_shared_contracts_and_only_the_current_platform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    golden_script: ModuleType,
    baseline: dict,
):
    current_key = "darwin-arm64-py311"
    other_key = "linux-x86_64-py311"
    stored = json.loads(json.dumps(baseline))
    current_snapshots = next(iter(stored["parse_by_platform"].values()))
    other_snapshots = json.loads(json.dumps(current_snapshots))
    other_snapshots["Doc1.pdf"]["chunks"][0]["content_sha256"] = "e" * 64
    stored["parse_by_platform"] = {
        current_key: current_snapshots,
        other_key: other_snapshots,
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(stored), encoding="utf-8")
    recorded = golden_script._selected_platform_contract(stored, current_key)

    monkeypatch.setattr(golden_script, "BASELINE", baseline_path)
    monkeypatch.setattr(golden_script, "REPO", tmp_path)
    monkeypatch.setattr(golden_script, "parser_platform_key", lambda: current_key)
    monkeypatch.setattr(
        golden_script,
        "_record_platform_contract",
        lambda platform_key: recorded,
    )

    assert golden_script.check() == 0


def test_golden_shape_has_no_application_or_provider_dependency(repo_root: Path):
    """A pure shape check must stay runnable with deliberately invalid config."""
    script = repo_root / "backend" / "scripts" / "record_golden.py"
    code = """
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("golden_contract", sys.argv[1])
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
shape = module.record_es_document_shape()
assert shape["vector_fields"] == ["q_1024_vec"]
assert shape["vector_dim"] == 1024
for forbidden in (
    "visionagent.config.settings",
    "visionagent.database.postgres.engine",
    "visionagent.database.elasticsearch.store",
    "visionagent.providers.embedding.online",
):
    assert forbidden not in sys.modules, forbidden
"""
    env = os.environ.copy()
    for name in (
        "APP_ENV", "JWT_SECRET_KEY", "DATABASE_URL", "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_DIM",
        "RERANK_MODEL", "CHAT_MODEL", "CHAT_MODEL_TURBO", "SERPER_API_KEY",
    ):
        env[name] = ""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------ analyzer
def test_analyzer_output_is_unchanged(baseline: dict):
    """Index-time and query-time analysis must produce identical strings.

    A drift here is silent: BM25 simply stops matching and retrieval quality
    degrades with no error anywhere.
    """
    from visionagent.vendor.ragflow.rag.nlp import rag_tokenizer

    drifted = []
    for text, expected in baseline["analyzer"].items():
        coarse = rag_tokenizer.tokenize(text)
        fine = rag_tokenizer.fine_grained_tokenize(coarse)
        if coarse != expected["content_tokens"]:
            drifted.append(
                f"{text!r} content_tokens: {expected['content_tokens']!r} -> {coarse!r}"
            )
        if fine != expected["content_tokens_fine"]:
            drifted.append(
                f"{text!r} content_tokens_fine: "
                f"{expected['content_tokens_fine']!r} -> {fine!r}"
            )
    assert not drifted, "analyzer output changed:\n  " + "\n  ".join(drifted)


def test_analyzer_still_stems(baseline: dict):
    """Pins that analysis is more than splitting -- it rewrites tokens.

    'distances' must not survive as 'distances'. If a replacement analyzer only
    splits, plural and inflected forms stop matching their stems.
    """
    from visionagent.vendor.ragflow.rag.nlp import rag_tokenizer

    out = rag_tokenizer.tokenize("Curb extensions reduce crossing distances")
    assert "extensions" not in out.split(), f"no stemming applied: {out!r}"
    assert "distances" not in out.split(), f"no stemming applied: {out!r}"


# --------------------------------------------------------------------- parse
@pytest.mark.slow
def test_parsed_documents_are_unchanged(repo_root: Path):
    """Chunking, OCR, layout grouping and analysis, end to end on real PDFs.

    Marked slow because it executes three real local OCR passes. The checker
    verifies source and model-asset hashes, starts one sanitized worker per
    fixed fixture, and compares the complete canonical snapshot. It uses no
    application configuration, datastore, or hosted provider.
    """
    script = repo_root / "backend" / "scripts" / "record_golden.py"
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        capture_output=True, text=True, timeout=1800,
    )
    tail = "\n".join(result.stdout.strip().splitlines()[-25:])
    assert result.returncode == 0, f"golden output changed:\n{tail}\n{result.stderr[-800:]}"


# ----------------------------------------------------------- es document shape
def test_es_document_field_names_are_unchanged(baseline: dict):
    """Typed chunk names must not change the persisted Elasticsearch schema."""
    # Field names are the contract; a zero-vector stub keeps this check offline.
    from visionagent.models import ParsedChunk
    from visionagent.service.vectorstore.elasticsearch import ElasticsearchChunkStore

    class StubAnalyzer:
        name = "stub"

        def analyze(self, text: str) -> str:
            raise AssertionError("document-shape test unexpectedly used analyzer")

        def analyze_fine(self, text: str) -> str:
            raise AssertionError("document-shape test unexpectedly used analyzer")

        def synonyms(self, term: str) -> list[str]:
            raise AssertionError("document-shape test unexpectedly used analyzer")

    class StubEmbedder:
        name, dimensions = "stub", 1024

        def embed(self, text: str) -> list[float]:
            return [0.0] * self.dimensions

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [self.embed(t) for t in texts]

        async def aembed(self, text: str) -> list[float]:
            return self.embed(text)

        async def aembed_batch(self, texts: list[str]) -> list[list[float]]:
            return self.embed_batch(texts)

        async def aclose(self) -> None:
            return None

    class RejectingWriter:
        def index(self, **kwargs):
            raise AssertionError("document-shape test attempted a write")

        def delete_document(self, **kwargs):
            raise AssertionError("document-shape test attempted a delete")

        def delete_index(self, **kwargs):
            raise AssertionError("document-shape test attempted a delete")

    store = ElasticsearchChunkStore(
        analyzer=StubAnalyzer(),
        embedder=StubEmbedder(),
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
    # The document shape is the store's; the typed parser chunk does not expose
    # persistent Elasticsearch field names.
    doc = store.to_document(chunk, index_name="golden-user", doc_name="golden.pdf")

    expected = baseline["es_document"]
    got = sorted(k for k in doc if not k.endswith("_vec"))
    assert got == expected["fields"], (
        "Elasticsearch field names changed. Existing indices are written with the "
        "old names and will stop matching.\n"
        f"  removed: {sorted(set(expected['fields']) - set(got))}\n"
        f"  added  : {sorted(set(got) - set(expected['fields']))}"
    )

    vec = sorted(k for k in doc if k.endswith("_vec"))
    assert vec == expected["vector_fields"], (
        f"vector field name changed: {expected['vector_fields']} -> {vec}; "
        "the mapping keys off q_<dim>_vec"
    )
    assert len(doc[vec[0]]) == expected["vector_dim"]


@pytest.mark.slow
def test_parse_is_reproducible_in_a_clean_process(
    repo_root: Path,
    baseline: dict,
    golden_script: ModuleType,
):
    """A fixed fixture must produce identical full snapshots in clean workers."""
    script = repo_root / "backend" / "scripts" / "record_golden.py"
    target = "Doc1.pdf"
    platform_key = golden_script.parser_platform_key()
    assert platform_key in baseline["parse_by_platform"], (
        f"missing parser baseline for {platform_key}; run make golden on that platform"
    )
    platform_baseline = baseline["parse_by_platform"][platform_key]
    assert target in platform_baseline

    runs = []
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, str(script), "--snapshot", target],
            capture_output=True, text=True, timeout=900,
        )
        assert result.returncode == 0, result.stderr[-500:]
        payloads = [
            line.removeprefix(SNAPSHOT_PREFIX)
            for line in result.stdout.splitlines()
            if line.startswith(SNAPSHOT_PREFIX)
        ]
        assert len(payloads) == 1, result.stdout[-500:]
        snapshot = json.loads(payloads[0])
        assert snapshot["chunk_count"] > 0
        runs.append(json.dumps(snapshot, sort_keys=True))

    assert len(set(runs)) == 1, (
        "the same document parsed differently across clean processes:\n  "
        + "\n  ".join(sorted(set(runs)))
    )
    assert json.loads(runs[0]) == platform_baseline[target]
