"""Private, bounded video intake for the local review application."""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import uuid

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def save_video_upload(upload, project_root: Path, probe=None, max_bytes: int = MAX_UPLOAD_BYTES) -> dict:
    """Store by content, verify decoding, and leave failed uploads unpublished."""
    root = Path(project_root).resolve()
    directory = (root / "data" / "uploads").resolve()
    directory.relative_to(root)
    filename = str(getattr(upload, "name", "video.mp4")).replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(filename).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError("Choose an MP4, MOV, MKV, AVI or WebM video.")
    if int(getattr(upload, "size", 0)) > max_bytes:
        raise ValueError("This video exceeds the 200 MB upload limit. Use a shorter clip.")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".incoming_{uuid.uuid4().hex}{suffix}"
    digest = hashlib.sha256()
    count = 0
    try:
        upload.seek(0)
        with temporary.open("wb") as handle:
            while chunk := upload.read(1024 * 1024):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("This video exceeds the upload limit. Use a shorter clip.")
                digest.update(chunk)
                handle.write(chunk)
        if not count:
            raise ValueError("The uploaded file is empty.")
        if probe is None:
            from .pipeline import video_info
            probe = video_info
        metadata = probe(temporary)
        for field in ("duration_seconds", "source_fps", "width", "height", "total_frames"):
            value = float(metadata.get(field, 0))
            if not math.isfinite(value) or value <= 0:
                raise ValueError("This file does not contain a readable video with a valid frame rate.")
        signature = digest.hexdigest()
        destination = directory / f"{signature}{suffix}"
        if destination.exists():
            # A truncated file must not be silently treated as the uploaded source.
            with destination.open("rb") as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != signature:
                    raise ValueError("A saved upload failed its content check. Choose another file.")
        else:
            os.replace(temporary, destination)
        return {"path": str(destination), "original_filename": filename,
                "sha256": signature, "size_bytes": count, "metadata": metadata,
                "source_type": "user_upload", "redistribution_permission": "not_requested"}
    finally:
        upload.seek(0)
        if temporary.exists():
            temporary.unlink()
