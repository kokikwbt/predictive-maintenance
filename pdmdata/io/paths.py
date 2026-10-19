"""Locate downloaded source files under the configured data root."""

from __future__ import annotations

from pathlib import Path

from pdmdata.config import get_settings


def find_raw_file(dataset_id: str, filename: str) -> Path:
    """Find a source file below the configured data root without downloading."""
    directory = get_settings().data_root / dataset_id
    matches = [path for path in directory.rglob(filename) if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"{filename} was not found under {directory}. Run pdmdata.download({dataset_id!r}) "
            "or place the dataset there first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple files named {filename} were found under {directory}"
        )
    return matches[0]
