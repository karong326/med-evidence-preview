"""Reconstruct local assets only from a trusted root and one plain filename."""

from pathlib import Path, PureWindowsPath
from uuid import UUID


def scoped_source(root: Path, owner: UUID | None, filename: str) -> Path:
    if (
        not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or PureWindowsPath(filename).name != filename
        or ":" in filename
        or PureWindowsPath(filename).is_reserved()
        or filename.endswith((" ", "."))
        or (filename.startswith(".") and filename.endswith(".part"))
    ):
        raise ValueError("Invalid source name")
    directory = root.resolve()
    if owner:
        directory = directory / str(owner)
    target = directory / filename
    if target.resolve() != target or directory.resolve() != directory:
        raise ValueError("Source escapes its registered location")
    return target
