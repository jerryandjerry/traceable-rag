import os
import re

MAX_CONTEXT_FILE_NAME_CHARS = 255
MAX_CONTEXT_FILE_NAME_BYTES = 255


def validate_context_filename(filename: str) -> None:
    """Bound a client name for both JSON metadata and filesystem components."""
    if not filename or len(filename) > MAX_CONTEXT_FILE_NAME_CHARS:
        raise ValueError("file names must contain 1-255 characters")
    if filename in {".", ".."} or "\x00" in filename or "/" in filename or "\\" in filename:
        raise ValueError("file name must be a single safe path component")
    sanitized = sanitize_filename(filename)
    if len(sanitized.encode("utf-8")) > MAX_CONTEXT_FILE_NAME_BYTES:
        raise ValueError("sanitized file name exceeds 255 UTF-8 bytes")


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe name while preserving its extension."""
    if not filename:
        return filename
    
    base_name, extension = os.path.splitext(filename)

    sanitized_base = re.sub(r'[^\w\s.-]', '', base_name)
    sanitized_base = re.sub(r'\s+', '_', sanitized_base)
    sanitized_base = re.sub(r'_+', '_', sanitized_base)
    sanitized_base = sanitized_base.strip('_')

    if not sanitized_base:
        sanitized_base = 'file'

    return sanitized_base + extension

def get_safe_file_path(directory: str, filename: str) -> tuple[str, str]:
    """Return a unique, sanitized name and its path within ``directory``."""
    validate_context_filename(filename)
    sanitized_name = sanitize_filename(filename)
    
    os.makedirs(directory, exist_ok=True)

    counter = 1
    original_name = sanitized_name
    base_name, extension = os.path.splitext(original_name)
    
    while os.path.exists(os.path.join(directory, sanitized_name)):
        sanitized_name = f"{base_name}_{counter}{extension}"
        if len(sanitized_name.encode("utf-8")) > MAX_CONTEXT_FILE_NAME_BYTES:
            raise ValueError("unique file name exceeds 255 UTF-8 bytes")
        counter += 1
    
    full_path = os.path.join(directory, sanitized_name)
    
    return sanitized_name, full_path
