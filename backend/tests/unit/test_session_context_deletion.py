from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from visionagent.database.session_context import SessionContextManager

SESSION_ID = "session-123"


@pytest.fixture
def manager(tmp_path: Path) -> SessionContextManager:
    context_dir = tmp_path / "uploads" / "session_context"
    context_dir.mkdir(parents=True)
    instance = SessionContextManager()
    instance.context_dir = str(context_dir)
    return instance


def _raw_directory(manager: SessionContextManager) -> Path:
    return Path(manager.context_dir).parent / "context_files" / SESSION_ID


def _entry(path: Path, name: str = "private.txt") -> dict[str, Any]:
    return {
        "file_name": name,
        "file_path": str(path),
        "content": "private content",
    }


def _save_entries(
    manager: SessionContextManager,
    entries: list[dict[str, Any]],
) -> Path:
    record = Path(manager._get_session_file(SESSION_ID))
    manager._save_session_context(
        SESSION_ID,
        {
            "session_id": SESSION_ID,
            "files": entries,
            "total_content": "private content",
            "created_at": "now",
            "updated_at": "now",
        },
    )
    return record


def _seed_file(manager: SessionContextManager) -> tuple[Path, Path]:
    raw = _raw_directory(manager) / "private.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("private bytes", encoding="utf-8")
    return raw, _save_entries(manager, [_entry(raw)])


def test_remove_keeps_retry_record_when_raw_erasure_fails(
    manager: SessionContextManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, record = _seed_file(manager)
    original_unlink = manager._unlink_raw

    def unavailable(_session_id: str, _file_path: str) -> None:
        raise OSError("volume unavailable")

    monkeypatch.setattr(manager, "_unlink_raw", unavailable)
    with pytest.raises(OSError, match="volume unavailable"):
        manager.remove_file_from_context(SESSION_ID, "private.txt")

    assert raw.exists()
    assert record.exists()
    assert manager._load_session_context(SESSION_ID)["files"]

    monkeypatch.setattr(manager, "_unlink_raw", original_unlink)
    assert manager.remove_file_from_context(SESSION_ID, "private.txt") is True
    assert not raw.exists()
    assert manager._load_session_context(SESSION_ID)["files"] == []


def test_remove_retry_tolerates_raw_erased_before_metadata_commit_failure(
    manager: SessionContextManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, record = _seed_file(manager)
    original_save = manager._save_session_context

    def full_disk(_session_id: str, _context: dict[str, Any]) -> None:
        raise OSError("metadata volume full")

    monkeypatch.setattr(manager, "_save_session_context", full_disk)
    with pytest.raises(OSError, match="metadata volume full"):
        manager.remove_file_from_context(SESSION_ID, "private.txt")

    assert not raw.exists()
    assert record.exists()
    assert manager._load_session_context(SESSION_ID)["files"]

    monkeypatch.setattr(manager, "_save_session_context", original_save)
    assert manager.remove_file_from_context(SESSION_ID, "private.txt") is True
    assert manager._load_session_context(SESSION_ID)["files"] == []


def test_clear_keeps_retry_record_until_raw_tree_erasure_succeeds(
    manager: SessionContextManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, record = _seed_file(manager)
    original_remove_tree = manager._remove_raw_session_directory

    def unavailable(_session_id: str) -> None:
        raise OSError("raw volume unavailable")

    monkeypatch.setattr(manager, "_remove_raw_session_directory", unavailable)
    with pytest.raises(OSError, match="raw volume unavailable"):
        manager.clear_session_context(SESSION_ID)

    assert raw.exists()
    assert record.exists()

    monkeypatch.setattr(
        manager,
        "_remove_raw_session_directory",
        original_remove_tree,
    )
    manager.clear_session_context(SESSION_ID)
    assert not raw.parent.exists()
    assert not record.exists()


def test_clear_retry_tolerates_raw_tree_erased_before_record_unlink_failure(
    manager: SessionContextManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, record = _seed_file(manager)
    original_unlink_record = manager._unlink_context_record

    def unavailable(_session_id: str) -> None:
        raise OSError("metadata volume unavailable")

    monkeypatch.setattr(manager, "_unlink_context_record", unavailable)
    with pytest.raises(OSError, match="metadata volume unavailable"):
        manager.clear_session_context(SESSION_ID)

    assert not raw.parent.exists()
    assert record.exists()

    monkeypatch.setattr(manager, "_unlink_context_record", original_unlink_record)
    manager.clear_session_context(SESSION_ID)
    assert not record.exists()


def test_clear_erases_orphans_without_following_child_symlinks(
    manager: SessionContextManager,
    tmp_path: Path,
) -> None:
    raw, record = _seed_file(manager)
    orphan = raw.parent / "orphan.tmp"
    orphan.write_text("orphaned private bytes", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "must-survive.txt"
    secret.write_text("do not delete", encoding="utf-8")
    (raw.parent / "outside-link").symlink_to(outside, target_is_directory=True)

    manager.clear_session_context(SESSION_ID)

    assert not raw.parent.exists()
    assert not record.exists()
    assert secret.read_text(encoding="utf-8") == "do not delete"


def test_clear_unlinks_a_session_directory_symlink_without_following_it(
    manager: SessionContextManager,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-session"
    outside.mkdir()
    secret = outside / "private.txt"
    secret.write_text("outside bytes", encoding="utf-8")
    raw_session = _raw_directory(manager)
    raw_session.parent.mkdir(parents=True)
    raw_session.symlink_to(outside, target_is_directory=True)
    record = _save_entries(manager, [_entry(raw_session / "private.txt")])

    manager.clear_session_context(SESSION_ID)

    assert not raw_session.exists()
    assert not raw_session.is_symlink()
    assert secret.read_text(encoding="utf-8") == "outside bytes"
    assert not record.exists()


def test_remove_unlinks_a_file_symlink_without_deleting_its_target(
    manager: SessionContextManager,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside bytes", encoding="utf-8")
    raw = _raw_directory(manager) / "private.txt"
    raw.parent.mkdir(parents=True)
    raw.symlink_to(outside)
    _save_entries(manager, [_entry(raw)])

    assert manager.remove_file_from_context(SESSION_ID, "private.txt") is True
    assert not raw.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside bytes"


@pytest.mark.parametrize("corruption", ["outside", "traversal"])
def test_corrupt_raw_path_fails_closed_before_any_deletion(
    manager: SessionContextManager,
    tmp_path: Path,
    corruption: str,
) -> None:
    owned = _raw_directory(manager) / "owned.txt"
    owned.parent.mkdir(parents=True)
    owned.write_text("owned bytes", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside bytes", encoding="utf-8")
    corrupt_path = (
        outside
        if corruption == "outside"
        else owned.parent / ".." / outside.name
    )
    record = _save_entries(manager, [_entry(corrupt_path)])

    with pytest.raises(ValueError, match="context file path"):
        manager.remove_file_from_context(SESSION_ID, "private.txt")

    assert owned.read_text(encoding="utf-8") == "owned bytes"
    assert outside.read_text(encoding="utf-8") == "outside bytes"
    assert record.exists()

    # Whole-session clear does not consume manifest paths. It erases only the
    # fixed owned directory, then the corrupt record, without touching the
    # external path named by that record.
    manager.clear_session_context(SESSION_ID)
    assert not owned.exists()
    assert outside.read_text(encoding="utf-8") == "outside bytes"
    assert not record.exists()


def test_clear_erases_owned_data_even_when_metadata_is_malformed(
    manager: SessionContextManager,
) -> None:
    raw = _raw_directory(manager) / "orphan.txt"
    raw.parent.mkdir(parents=True)
    raw.write_text("private bytes", encoding="utf-8")
    record = Path(manager._get_session_file(SESSION_ID))
    record.write_text("{not valid JSON", encoding="utf-8")

    manager.clear_session_context(SESSION_ID)

    assert not raw.parent.exists()
    assert not record.exists()


def test_remove_returns_false_only_for_an_absent_name_in_a_valid_record(
    manager: SessionContextManager,
) -> None:
    raw, _record = _seed_file(manager)

    assert manager.remove_file_from_context(SESSION_ID, "not-attached.txt") is False
    assert raw.exists()


def test_remove_succeeds_when_the_owned_raw_file_is_already_missing(
    manager: SessionContextManager,
) -> None:
    raw = _raw_directory(manager) / "private.txt"
    record = _save_entries(manager, [_entry(raw)])

    assert manager.remove_file_from_context(SESSION_ID, "private.txt") is True
    assert record.exists()
    assert manager._load_session_context(SESSION_ID)["files"] == []


def test_traversal_session_id_cannot_select_a_raw_or_metadata_target(
    manager: SessionContextManager,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside bytes", encoding="utf-8")

    with pytest.raises(ValueError, match="session id"):
        manager.clear_session_context("../outside")

    assert outside.read_text(encoding="utf-8") == "outside bytes"
