#!/usr/bin/env python3
"""Generate the ALPI raw-event inspection notebook without downloading data."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    item = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


def main():
    cells = [
        cell("markdown", """# ALPI packaging-machine alarms

Inspect original event times, alarm codes, and machine identifiers. This notebook
does not create failure labels, prediction windows, or train/test splits.

First download with `uv run --locked python scripts/download.py alpi` from the
project root. Loading below reads local files only. Source: [Mendeley Data,
version 1](https://data.mendeley.com/datasets/4nhx2x67cd/1), CC BY 4.0.
"""),
        cell("code", """import pdmdata
from pdmdata.datasets.alpi import inventory, validate

inventory()
"""),
        cell("markdown", """## Original events

Keep source order and duplicate events. Raw alarm code 0 is genuine; it is not
padding. `source_row` identifies each original CSV row after filtering. Provider
timestamps do not specify a timezone. The last alarm is not a failure endpoint.
"""),
        cell("code", """events = pdmdata.load("alpi", serial=4)
events.head(10).collect()
"""),
        cell("markdown", """## Source validation

Report field validity, duplicates, and per-machine ordering without deleting or
reordering observations. The 2026-10-18 full-file check covered 444,834 events
and matched the provider archive checksum and an independent CSV reader.
"""),
        cell("code", "validate()\n"),
    ]
    document = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path = ROOT / "notebooks/datasets/alpi.ipynb"
    path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
