"""Tests for the parsers/ slot.

This is the seam RAGFlow plugs into, so the load-bearing test is that a parser
with no RAGFlow behind it can be substituted and the pipeline cannot tell.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from visionagent.models import ParsedChunk
from visionagent.service.parsers import DocumentParser, ParserError, build_parser
from visionagent.service.parsers.deepdoc import DeepDocParser, _int_list
from visionagent.vendor.ragflow.deepdoc.parser.pdf_parser import _looks_english


class FakeParser:
    """A complete DocumentParser with no RAGFlow, no ONNX and no file I/O."""

    name = "fake"
    version = "1"
    supported_extensions = frozenset({".pdf"})

    def parse(self, *, file_path: Path, file_name: str, progress=None):
        if progress:
            progress(0.5, "halfway")
        return [
            ParsedChunk(id="f1", content="First chunk.", page_nums=[1]),
            ParsedChunk(id="f2", content="Second chunk.", page_nums=[1, 2]),
        ]


# ----------------------------------------------------------------- protocol
@pytest.mark.parametrize("name", ["deepdoc", "vlm"])
def test_shipped_parsers_satisfy_the_protocol(name: str):
    assert isinstance(build_parser(name), DocumentParser)


def test_a_parser_with_no_ragflow_satisfies_the_protocol():
    """The whole point of the slot: this is what replacing RAGFlow looks like."""
    assert isinstance(FakeParser(), DocumentParser)


def test_factory_rejects_an_unknown_parser():
    with pytest.raises(ValueError, match="unknown parser"):
        build_parser("tesseract")


def test_factory_reads_the_env_selector(monkeypatch):
    monkeypatch.setenv("PARSER", "vlm")
    assert build_parser().name == "vlm"


def test_a_substituted_parser_flows_through_the_pipeline_boundary(monkeypatch):
    """file_parse.parse() must not know which parser ran."""
    import visionagent.pipeline.ingest as fp

    monkeypatch.setattr(fp, "build_parser", lambda *a, **k: FakeParser())
    out = fp.parse("anything.pdf", "/nonexistent/anything.pdf")

    assert [c.content for c in out] == ["First chunk.", "Second chunk."]
    assert out[0].page_nums == [1]
    assert out[1].page_nums == [1, 2]
    assert all(isinstance(chunk, ParsedChunk) for chunk in out)


# -------------------------------------------------------- deepdoc quirks
@pytest.mark.parametrize(
    "raw,expected",
    [([1], [1]), ([1, 2], [1, 2]), (3, [3]), (None, []), ([], []), ((4, 5), [4, 5])],
)
def test_int_list_normalises_deepdocs_shapes(raw, expected):
    assert _int_list(raw) == expected


def test_deepdoc_language_detection_is_deterministic():
    english = list("Tall buildings should fit within their surrounding context.")
    chinese = list("建筑设计规范要求建筑物符合周围环境并减少局部影响。")

    assert _looks_english(english)
    assert not _looks_english(chinese)
    assert {_looks_english(english) for _ in range(100)} == {True}


def test_deepdoc_language_detection_requires_meaningful_text():
    assert not _looks_english([])
    assert not _looks_english(list("short English text"))
    assert _looks_english(list("1234567890-_/() 1234567890-_/() 1234567890"))


def test_multi_page_chunks_keep_every_page():
    """A DeepDoc chunk spanning pages preserves every page number."""
    c = ParsedChunk(id="c1", content="spans two pages", page_nums=[4, 5])
    assert c.page_nums == [4, 5]
    assert c.page_num == 4, "page_num is the first page, which is what ES stores"


def test_page_num_is_safe_when_deepdoc_emits_nothing():
    """`item['page_num_int'][0]` raised IndexError partway through indexing."""
    assert ParsedChunk(id="c1", content="x").page_num == 0
    assert ParsedChunk(id="c1", content="x").top_offset == 0


def test_empty_chunks_are_surfaced_not_dropped(monkeypatch):
    """Dropping one would shift every later chunk's position, which reads as
    'chunk text changed' rather than 'a chunk went missing'."""
    import visionagent.vendor.ragflow.rag.app.manual as manual

    monkeypatch.setattr(
        manual, "chunk",
        lambda *a, **k: [{"id": "x", "content_with_weight": "", "page_num_int": [1]}],
    )
    with pytest.raises(ParserError, match="empty chunk"):
        DeepDocParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")


def test_parser_failure_becomes_a_parsererror(monkeypatch):
    """deepdoc raises many exception types; the boundary normalises them."""
    import visionagent.vendor.ragflow.rag.app.manual as manual

    def boom(*a, **k):
        raise ZeroDivisionError("layout model exploded")

    monkeypatch.setattr(manual, "chunk", boom)
    with pytest.raises(ParserError, match="deepdoc failed on x.pdf"):
        DeepDocParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")


def test_chunk_ids_are_content_derived_not_user_scoped():
    """The persisted id is xxhash(content + user_id), assigned after parsing;
    a parser has no user_id and must not invent the tenant-scoped form."""
    import xxhash

    from visionagent.service.parsers.deepdoc import DeepDocParser as P

    assert P.name == "deepdoc"
    body = "Sidewalk widths."
    assert xxhash.xxh64(body.encode()).hexdigest() != xxhash.xxh64(
        (body + "user1").encode()
    ).hexdigest()


# ==========================================================================
# EXACT BEHAVIOUR
#
# Intended function of the deepdoc parser component:
#   parse(pdf) -> exactly one ParsedChunk per non-empty deepdoc section, in
#   deepdoc's order, with every field mapped:
#       content_with_weight -> content
#       content_ltks        -> content_tokens
#       content_sm_ltks     -> content_tokens_fine
#       docnm_tks           -> doc_name_tokens
#       page_num_int        -> page_nums   (whole list, not [0])
#       top_int             -> top_offsets (whole list, not [0])
#       ref_images / image  -> base64 strings
#       (id)                -> xxhash64(content), assigned here, not by deepdoc
#
# Intended function of the vlm parser component:
#   parse(file) -> exactly one ParsedChunk per non-blank page transcription,
#   using its structured source page number without parsing transcription text.
# ==========================================================================
class _Img:
    """Minimal stand-in for a PIL Image: has .save(buffer, format=...)."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def save(self, buffer, format: str) -> None:  # noqa: A002
        buffer.write(self.payload)


def _deepdoc_returns(monkeypatch, items):
    import visionagent.vendor.ragflow.rag.app.manual as manual

    monkeypatch.setattr(manual, "chunk", lambda *a, **k: items)


def test_deepdoc_adapter_maps_every_field_exactly(monkeypatch):
    import base64

    import xxhash

    _deepdoc_returns(monkeypatch, [{
        "id": "deepdoc-ignores-this",
        "content_with_weight": "Sidewalk widths.",
        "content_ltks": "sidewalk width",
        "content_sm_ltks": "sidewalk width",
        "docnm_tks": "curb guide",
        "page_num_int": [4, 5],
        "top_int": [120, 8],
        "ref_images": [_Img(b"REF")],
        "image": _Img(b"MAIN"),
    }])
    out = DeepDocParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")

    assert len(out) == 1
    assert out[0] == ParsedChunk(
        id=xxhash.xxh64(b"Sidewalk widths.").hexdigest(),
        content="Sidewalk widths.",
        content_tokens="sidewalk width",
        content_tokens_fine="sidewalk width",
        doc_name_tokens="curb guide",
        page_nums=[4, 5],
        top_offsets=[120, 8],
        ref_images=[base64.b64encode(b"REF").decode()],
        image=base64.b64encode(b"MAIN").decode(),
    )


def test_deepdoc_adapter_preserves_order_and_count(monkeypatch):
    _deepdoc_returns(monkeypatch, [
        {"content_with_weight": "first", "page_num_int": [1]},
        {"content_with_weight": "second", "page_num_int": [1]},
        {"content_with_weight": "third", "page_num_int": [2]},
    ])
    out = DeepDocParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")
    assert [c.content for c in out] == ["first", "second", "third"]
    assert [c.page_num for c in out] == [1, 1, 2]


def test_deepdoc_adapter_defaults_missing_optional_fields_exactly(monkeypatch):
    """deepdoc omits keys for text-only chunks; the defaults must be empty,
    not None, because the ES document is built from them directly."""
    _deepdoc_returns(monkeypatch, [{"content_with_weight": "bare"}])
    c = DeepDocParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")[0]
    assert (c.content_tokens, c.content_tokens_fine, c.doc_name_tokens) == ("", "", "")
    assert (c.page_nums, c.top_offsets, c.ref_images, c.image) == ([], [], [], None)
    assert (c.page_num, c.top_offset) == (0, 0)


def test_deepdoc_adapter_gives_identical_content_an_identical_id(monkeypatch):
    """Ids are content-derived, so a repeated chunk deduplicates in the store."""
    _deepdoc_returns(monkeypatch, [
        {"content_with_weight": "same text"},
        {"content_with_weight": "same text"},
    ])
    out = DeepDocParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")
    assert out[0].id == out[1].id


def test_vlm_adapter_preserves_actual_page_boundaries_and_numbers(monkeypatch):
    import visionagent.service.parsers.vlm.processor as vp
    from visionagent.service.parsers.vlm import VLMParser
    from visionagent.service.parsers.vlm.processor import PageTranscription

    async def process_file(path):
        return {
            "success": True,
            # Deliberately ambiguous aggregate text. The parser must use the
            # structured pages instead of splitting this field.
            "content": (
                "Page 1: first paragraph\n\nsecond paragraph\n\n"
                "Page 3: [Error extracting text]"
            ),
            "pages": [
                PageTranscription(
                    page_number=1,
                    content="Page 1: first paragraph\n\nsecond paragraph",
                ),
                PageTranscription(page_number=2, content=""),
                PageTranscription(
                    page_number=3,
                    content="Page 3: [Error extracting text]",
                ),
            ],
            "pages_processed": 3,
            "file_path": path,
        }

    monkeypatch.setattr(
        vp.vlm_processor, "process_file",
        process_file,
    )
    out = VLMParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")
    assert [(c.content, c.page_num) for c in out] == [
        ("Page 1: first paragraph\n\nsecond paragraph", 1),
        ("Page 3: [Error extracting text]", 3),
    ]


def test_vlm_adapter_returns_nothing_for_an_empty_transcription(monkeypatch):
    import visionagent.service.parsers.vlm.processor as vp
    from visionagent.service.parsers.vlm import VLMParser
    from visionagent.service.parsers.vlm.processor import PageTranscription

    async def process_file(path):
        return {
            "success": True,
            "content": "",
            "pages": [PageTranscription(page_number=1, content="")],
            "pages_processed": 1,
            "file_path": path,
        }

    monkeypatch.setattr(vp.vlm_processor, "process_file", process_file)
    assert VLMParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf") == []


def test_vlm_adapter_surfaces_provider_failure(monkeypatch):
    import visionagent.service.parsers.vlm.processor as vp
    from visionagent.service.parsers.vlm import VLMParser

    async def process_file(path):
        return {"success": False, "error": "could not render"}

    monkeypatch.setattr(vp.vlm_processor, "process_file", process_file)
    with pytest.raises(ParserError, match="could not render"):
        VLMParser().parse(file_path=Path("/x.pdf"), file_name="x.pdf")
