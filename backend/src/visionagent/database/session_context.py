import contextlib
import json
import logging
import os
import shutil
import stat
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from visionagent.config.settings import settings

logger = logging.getLogger(__name__)


class SessionContextManager:
    def __init__(self, *, storage_dir: Path | None = None) -> None:
        root = settings.storage_dir if storage_dir is None else storage_dir
        self.context_dir = str(root / "session_context")
        os.makedirs(self.context_dir, exist_ok=True)
    
    def _get_session_file(self, session_id: str) -> str:
        """Get the file path for storing session context"""
        self._validate_component(session_id, "session id")
        return os.path.join(self.context_dir, f"{session_id}_context.json")
    
    def _load_session_context(self, session_id: str) -> dict[str, Any]:
        """Load existing session context from file"""
        file_path = self._get_session_file(session_id)
        try:
            with open(file_path, encoding='utf-8') as f:
                return cast(dict[str, Any], json.load(f))
        except FileNotFoundError:
            pass
        return {
            "session_id": session_id,
            "files": [],
            "total_content": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
    
    def _save_session_context(self, session_id: str, context: dict[str, Any]) -> None:
        """Save session context to file.

        Writes are fsynced and atomically renamed over the target. Failures
        propagate so callers never report content as attached before it is
        durable.
        """
        file_path = self._get_session_file(session_id)
        context["updated_at"] = datetime.now().isoformat()
        fd, tmp = tempfile.mkstemp(dir=self.context_dir, prefix=f"{session_id}_", suffix=".tmp")
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(context, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, file_path)
            directory_fd = os.open(self.context_dir, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
    
    def add_content_to_context(self, session_id: str, file_path: str, file_name: str,
                               content: str, pages_processed: int = 0) -> dict[str, Any]:
        """Store already-extracted text against a session.

        ``pipeline.context`` owns extraction; storage persists the supplied
        content without invoking a parser or model.
        """
        try:
            context = self._load_session_context(session_id)
            self._context_files(session_id, context)
            self._validated_raw_name(session_id, file_path)

            file_info = {
                "file_name": file_name,
                "file_path": file_path,
                "content": content,
                "pages_processed": pages_processed,
                "added_at": datetime.now().isoformat(),
                "processing_time": time.time()  # Will be updated with actual time
            }
            
            context["files"].append(file_info)
            
            if context["total_content"]:
                context["total_content"] += "\n\n--- New File: " + file_name + " ---\n\n"
            context["total_content"] += content
            
            self._save_session_context(session_id, context)
            
            return {
                "success": True,
                "file_name": file_name,
                "content_length": len(content),
                "pages_processed": pages_processed,
                "total_files": len(context["files"])
            }
            
        except Exception as e:
            logger.exception("session-context update failed")
            return {
                "success": False,
                "error": str(e),
                "file_name": file_name
            }
    
    def get_session_context(self, session_id: str) -> str:
        """
        Get the total context content for a session
        """
        context = self._load_session_context(session_id)
        return str(context.get("total_content", ""))
    
    def get_session_files(self, session_id: str) -> list[dict[str, Any]]:
        """
        Get list of files in session context
        """
        context = self._load_session_context(session_id)
        return cast(list[dict[str, Any]], context.get("files", []))
    
    def clear_session_context(self, session_id: str) -> None:
        """Durably erase a session's raw files before its metadata record.

        The JSON record remains the retry handle until raw erasure succeeds.
        Removing the whole owned directory also clears raw files left behind by
        a crash before their metadata commit. Clearing never reads deletion
        targets from the manifest: even malformed or legacy JSON cannot block
        erasure, and no corrupt path can steer it outside the session-owned
        directory.
        """
        self._remove_raw_session_directory(session_id)
        self._unlink_context_record(session_id)
        logger.info("session context cleared")
    
    def remove_file_from_context(self, session_id: str, file_name: str) -> bool:
        """Erase matching raw bytes, then atomically remove their metadata."""
        context = self._load_session_context(session_id)
        files = self._context_files(session_id, context)

        removed = [item for item in files if item["file_name"] == file_name]
        retained = [item for item in files if item["file_name"] != file_name]

        if not removed:
            return False

        # Leave the old JSON intact if any unlink fails. A retry tolerates raw
        # files that this attempt already removed and resumes from the record.
        for file_info in removed:
            self._unlink_raw(session_id, str(file_info["file_path"]))

        context["files"] = retained
        context["total_content"] = ""
        for file_info in retained:
            if context["total_content"]:
                context["total_content"] += (
                    "\n\n--- File: " + file_info["file_name"] + " ---\n\n"
                )
            context["total_content"] += str(file_info["content"])

        self._save_session_context(session_id, context)
        return True

    @staticmethod
    def _validate_component(value: str, label: str) -> None:
        if (
            not isinstance(value, str)
            or not value
            or value in {".", ".."}
            or "\x00" in value
            or "/" in value
            or "\\" in value
        ):
            raise ValueError(f"{label} must be one safe path component")

    def _raw_root(self) -> str:
        # context_dir is <storage>/session_context in production. Deriving its
        # sibling keeps tests and alternate STORAGE_DIR deployments isolated.
        return os.path.abspath(
            os.path.join(os.path.dirname(self.context_dir), "context_files")
        )

    def _validated_raw_name(self, session_id: str, file_path: str) -> str:
        """Return a direct-child name without resolving the final symlink.

        Resolution would follow a malicious final symlink. Instead, deletion
        later opens the raw root and session directory with O_NOFOLLOW and
        unlinks this single directory entry relative to those descriptors.
        """
        self._validate_component(session_id, "session id")
        if not isinstance(file_path, str) or "\x00" in file_path:
            raise ValueError("context file path is invalid")
        if not os.path.isabs(file_path) or os.path.normpath(file_path) != file_path:
            raise ValueError("context file path is not a canonical absolute path")

        session_directory = os.path.join(self._raw_root(), session_id)
        if os.path.dirname(file_path) != session_directory:
            raise ValueError("context file path is outside its session directory")

        name = os.path.basename(file_path)
        self._validate_component(name, "context file name")
        return name

    def _context_files(
        self,
        session_id: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Validate the persisted deletion manifest before changing anything."""
        files = context.get("files")
        if not isinstance(files, list):
            raise ValueError("session context files must be a list")
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("session context file entry must be an object")
            file_name = item.get("file_name")
            file_path = item.get("file_path")
            if not isinstance(file_name, str) or not isinstance(file_path, str):
                raise ValueError("session context file entry is incomplete")
            self._validated_raw_name(session_id, file_path)
        return cast(list[dict[str, Any]], files)

    @staticmethod
    def _directory_open_flags() -> int:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        return flags

    def _open_raw_root(self) -> int | None:
        try:
            return os.open(self._raw_root(), self._directory_open_flags())
        except FileNotFoundError:
            return None

    def _remove_raw_session_directory(self, session_id: str) -> None:
        """Remove only the owned tree, without following directory symlinks."""
        self._validate_component(session_id, "session id")
        root_descriptor = self._open_raw_root()
        if root_descriptor is None:
            return
        try:
            try:
                entry = os.stat(
                    session_id,
                    dir_fd=root_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return

            if stat.S_ISDIR(entry.st_mode):
                if not shutil.rmtree.avoids_symlink_attacks:
                    raise RuntimeError("safe directory-tree deletion is unavailable")
                shutil.rmtree(session_id, dir_fd=root_descriptor)
            else:
                # A symlink (including one to a directory) is unlinked as an
                # entry under raw_root; its target is never traversed.
                os.unlink(session_id, dir_fd=root_descriptor)
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)

    def _unlink_context_record(self, session_id: str) -> None:
        file_path = self._get_session_file(session_id)
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            return
        self._fsync_directory(self.context_dir)

    @staticmethod
    def _fsync_directory(directory: str) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _unlink_raw(self, session_id: str, file_path: str) -> None:
        name = self._validated_raw_name(session_id, file_path)
        root_descriptor = self._open_raw_root()
        if root_descriptor is None:
            return
        try:
            try:
                session_descriptor = os.open(
                    session_id,
                    self._directory_open_flags(),
                    dir_fd=root_descriptor,
                )
            except FileNotFoundError:
                return
            try:
                try:
                    os.unlink(name, dir_fd=session_descriptor)
                except FileNotFoundError:
                    return
                os.fsync(session_descriptor)
            finally:
                os.close(session_descriptor)
        finally:
            os.close(root_descriptor)

session_context_manager = SessionContextManager()
