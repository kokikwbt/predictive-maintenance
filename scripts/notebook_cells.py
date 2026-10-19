"""Shared cell builders and writer for the notebook generator scripts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

METADATA = {
    "kernelspec": {
        "display_name": "Python (pdmdata)",
        "language": "python",
        "name": "pdmdata",
    },
    "language_info": {"name": "python", "version": "3.14"},
}


def markdown(source: str, identifier: str | None = None) -> dict:
    """Create a Markdown cell; ``identifier`` defaults to its position."""
    cell = {"cell_type": "markdown", "metadata": {}}
    if identifier is not None:
        cell["id"] = identifier
    cell["source"] = source.splitlines(True)
    return cell


def code(source: str, identifier: str | None = None) -> dict:
    """Create an unexecuted code cell; ``identifier`` defaults to position."""
    cell = markdown(source, identifier)
    cell.update(cell_type="code", execution_count=None, outputs=[])
    return cell


def notebook(cells: list[dict]) -> dict:
    """Wrap cells in a project-kernel document, numbering cells without ids."""
    for index, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{index:02d}")
    return {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_document(path: Path, document: dict) -> Path:
    """Write a notebook document as stable JSON and print the relative path."""
    path = Path(path)
    path.write_text(
        json.dumps(document, indent=1, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    print(path.relative_to(ROOT))
    return path


def write_notebook(path: Path, cells: list[dict]) -> Path:
    """Write ``cells`` as a project-kernel notebook at ``path``."""
    return write_document(path, notebook(cells))


class NotebookBuilder:
    """Collect cells in order, then write them with ``write``."""

    def __init__(self) -> None:
        self.cells: list[dict] = []

    def markdown(self, source: str) -> None:
        self.cells.append(markdown(source))

    def code(self, source: str) -> None:
        self.cells.append(code(source))

    def write(self, path: Path) -> Path:
        return write_notebook(path, self.cells)
