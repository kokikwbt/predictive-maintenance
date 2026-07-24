"""Load machine-readable metadata for the bundled datasets."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterator, List, Optional, TextIO


ROOT = Path(__file__).resolve().parent


def metadata_paths() -> Iterator[Path]:
    """Yield metadata files in stable dataset-id order."""
    yield from sorted(ROOT.glob("*/metadata.json"), key=lambda path: path.parent.name)


def load_metadata(dataset_id: str) -> Dict[str, Any]:
    """Load metadata for one dataset."""
    path = ROOT / dataset_id / "metadata.json"
    if not path.is_file():
        raise KeyError("Unknown dataset: {!r}".format(dataset_id))
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("id") != dataset_id:
        raise ValueError("{}: id must be {!r}".format(path, dataset_id))
    return metadata


def load_catalog(
    *,
    task: Optional[str] = None,
    available: Optional[bool] = None,
    downloadable: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Load all dataset metadata, optionally filtering common properties."""
    catalog = [load_metadata(path.parent.name) for path in metadata_paths()]
    if task is not None:
        catalog = [item for item in catalog if task in item.get("tasks", [])]
    if available is not None:
        expected = "available" if available else "unavailable"
        catalog = [
            item
            for item in catalog
            if item.get("availability", {}).get("status") == expected
        ]
    if downloadable is not None:
        catalog = [
            item for item in catalog if ("download" in item) is downloadable
        ]
    return catalog


def format_summary(*, downloadable: Optional[bool] = None) -> str:
    """Return a plain-text table summarizing the dataset catalog."""
    rows = []
    for item in load_catalog(downloadable=downloadable):
        rows.append(
            [
                item["id"],
                item["short_name"],
                item["name"],
                item["availability"]["status"],
                "yes" if "download" in item else "no",
                item["license"]["name"],
                ", ".join(item.get("tasks", [])),
            ]
        )

    headers = [
        "ID",
        "Short name",
        "Dataset",
        "Availability",
        "Download",
        "License",
        "Tasks",
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: List[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        ).rstrip()

    separator = ["-" * width for width in widths]
    return "\n".join([render(headers), render(separator)] + [render(row) for row in rows])


def summary(
    *,
    downloadable: Optional[bool] = None,
    file: Optional[TextIO] = None,
) -> None:
    """Print the available dataset catalog.

    Set ``downloadable=True`` to show datasets supported by direct URLs or the
    official Kaggle CLI, or ``downloadable=False`` to show unsupported datasets.
    """
    print(format_summary(downloadable=downloadable), file=file or sys.stdout)
