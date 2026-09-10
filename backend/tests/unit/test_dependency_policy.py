"""Machine-checked conditions attached to temporary dependency exceptions."""

from pathlib import Path


def test_nltk_advisory_exception_remains_inapplicable(app_dir: Path) -> None:
    """The ignored path-traversal advisory covers APIs this project must not use."""
    affected_apis = {
        "TransitionParser",
        "AveragedPerceptron",
        "PerceptronTagger",
        "save_maxent_params",
    }
    offenders: list[str] = []
    for path in app_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        used = sorted(name for name in affected_apis if name in source)
        if used:
            offenders.append(f"{path.relative_to(app_dir)}: {', '.join(used)}")

    assert not offenders, (
        "PYSEC-2026-3740 may now apply; remove the CI waiver or isolate the model path:\n"
        + "\n".join(offenders)
    )


def test_bundled_parser_never_downloads_models_at_runtime(app_dir: Path) -> None:
    """Model identity is fixed by Git LFS and the golden asset manifest."""
    loaders = (
        "vendor/ragflow/deepdoc/parser/pdf_parser.py",
        "vendor/ragflow/deepdoc/vision/layout_recognizer.py",
        "vendor/ragflow/deepdoc/vision/ocr.py",
        "vendor/ragflow/deepdoc/vision/recognizer.py",
        "vendor/ragflow/deepdoc/vision/table_structure_recognizer.py",
    )
    offenders = [
        relative
        for relative in loaders
        if "snapshot_download" in (app_dir / relative).read_text(encoding="utf-8")
    ]
    assert not offenders, (
        "parser model loaders must fail on missing assets instead of mutating the "
        f"installation at runtime: {', '.join(offenders)}"
    )
