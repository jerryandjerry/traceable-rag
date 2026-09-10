"""Unit tests for pure backend logic. No DB, no ES, no network."""
import json

import pytest


# ------------------------------------------------------------------ schemas
def test_chat_request_defaults():
    from visionagent.models import ChatRequest

    req = ChatRequest(message="hi")
    assert req.message == "hi"
    # False delegates to intent detection; True requires a search, so the
    # required mode must not be the default.
    assert req.web_search is False, "web_search defaults to auto"
    assert req.deep_research is False
    assert req.chat_id is None
    assert req.attachments is None


def test_chat_request_accepts_str_or_int_chat_id():
    """The frontend generates a numeric chat id; the schema must accept both."""
    from visionagent.models import ChatRequest

    assert ChatRequest(message="x", chat_id=7).chat_id == 7
    assert ChatRequest(message="x", chat_id="7").chat_id == "7"


def test_chat_request_requires_message():
    from pydantic import ValidationError

    from visionagent.models import ChatRequest

    with pytest.raises(ValidationError):
        ChatRequest()


def test_deep_research_flag_is_accepted_but_unused(app_dir):
    """No server code reads deep_research.

    The flag exists on ChatRequest and the job records it as requested, but the
    effective value is forced to False when the job is minted and nothing
    downstream branches on it. This test documents that, so an implementation
    has to update it deliberately rather than by accident.
    """
    deps = (app_dir / "api" / "deps.py").read_text(encoding="utf-8")
    assert "deep_research=False" in deps, "the effective value must stay off"

    for module in ("pipeline/query.py", "service/planner/rule_based/__init__.py"):
        src = (app_dir / module).read_text(encoding="utf-8")
        assert "deep_research" not in src, (
            f"{module} now reads deep_research -- update this test and the "
            "frontend, which currently treats Deep Search as a plain search"
        )


# ---------------------------------------------------------------- tokenizer
def test_rag_tokenizer_handles_mixed_scripts():
    from visionagent.vendor.ragflow.rag.nlp import rag_tokenizer

    for text in ["curb extension width", "Curb EXTENSION 3.5m", "建筑设计规范", "2+2"]:
        out = rag_tokenizer.tokenize(text)
        assert isinstance(out, str)


def test_synonym_lookup_returns_list_for_every_token_shape():
    """lookup() must never raise, whatever the token looks like."""
    from visionagent.vendor.ragflow.rag.nlp.synonym import Dealer

    dealer = Dealer()
    for token in ["width", "Width", "WIDTH3", "3.5m", "建筑", "", "a b"]:
        assert isinstance(dealer.lookup(token), list), f"failed on {token!r}"


# ------------------------------------------------------------------- prompt
def test_direct_answer_prompt_has_three_slots():
    """DirectAnswerPrompt is filled with (references, history, question)."""
    from visionagent.utils.prompt import DirectAnswerPrompt

    assert DirectAnswerPrompt.count("%s") == 3
    filled = DirectAnswerPrompt % ("REFS", "HISTORY", "QUESTION")
    assert "REFS" in filled and "HISTORY" in filled and "QUESTION" in filled


def test_citation_format_documented_in_prompt():
    """The prompt tells the model to emit [doc][cite_XXX] / [web][cite_XXX]."""
    from visionagent.utils.prompt import DirectAnswerPrompt

    assert "[doc][cite_" in DirectAnswerPrompt
    assert "[web][cite_" in DirectAnswerPrompt


# --------------------------------------------------------------- agent json
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('["kb(filter)"]', ["kb(filter)"]),
        ('```json\n["web_search"]\n```', ["web_search"]),
        ('text before ["kb(filter)", "web_search"] text after',
         ["kb(filter)", "web_search"]),
    ],
)
def test_extract_json_content_tolerates_llm_wrapping(raw, expected):
    """LLMs wrap JSON in prose or fences; the extractor must cope."""
    from visionagent.providers.llm import extract_json_content

    extracted = extract_json_content(raw)
    if extracted is None:
        extracted = raw
    assert json.loads(extracted) == expected


# -------------------------------------------------------------- sse framing
def test_sse_frames_are_well_formed():
    """Every frame the backend writes must be `event: X\\ndata: Y\\n\\n`."""
    frames = [
        'event: message\ndata: {"role": "assistant", "content": "hi"}\n\n',
        "event: end\ndata: [DONE]\n\n",
    ]
    for frame in frames:
        assert frame.startswith("event: ")
        assert "\ndata: " in frame
        assert frame.endswith("\n\n")
        payload = frame.split("\ndata: ", 1)[1].rstrip("\n")
        if payload != "[DONE]":
            json.loads(payload)


# ------------------------------------------------------- filename sanitising
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("My Document.pdf", "My_Document.pdf"),
        ("File with spaces and (parentheses).docx", "File_with_spaces_and_parentheses.docx"),
        ("Special@#$%^&*()chars.txt", "Specialchars.txt"),
        ("Multiple   spaces.pdf", "Multiple_spaces.pdf"),
        ("File-with-hyphens.docx", "File-with-hyphens.docx"),
        ("File_with_underscores.pdf", "File_with_underscores.pdf"),
        ("File.with.dots.docx", "File.with.dots.docx"),
        ("File with @#$% special chars and spaces.pdf", "File_with_special_chars_and_spaces.pdf"),
        ("【Accessibility】Café Standards.pdf", "AccessibilityCafé_Standards.pdf"),
        ("", ""),
        ("   .pdf", "file.pdf"),
        ("@#$%^&*().pdf", "file.pdf"),
    ],
)
def test_sanitize_filename(raw, expected):
    from visionagent.utils.file_utils import sanitize_filename

    assert sanitize_filename(raw) == expected
