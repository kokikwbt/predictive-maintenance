#!/usr/bin/env python3
"""Generate the ALPI raw-event inspection notebook without downloading data."""

from notebook_cells import ROOT, code, markdown, write_notebook


def main():
    cells = [
        markdown("""# ALPI packaging-machine alarms

Inspect original event times, alarm codes, and machine identifiers. This notebook
does not create failure labels, prediction windows, or train/test splits.

First download with `uv run --locked python scripts/download.py alpi` from the
project root. Loading below reads local files only. Source: [Mendeley Data,
version 1](https://data.mendeley.com/datasets/4nhx2x67cd/1), CC BY 4.0.
"""),
        code("""import pdmdata
from pdmdata.datasets.alpi import inventory, verify

inventory()
"""),
        markdown("""## Original events

Keep source order and duplicate events. Raw alarm code 0 is genuine; it is not
padding. `source_row` identifies each original CSV row after filtering. Provider
timestamps do not specify a timezone. The last alarm is not a failure endpoint.
"""),
        code("""events = pdmdata.load("alpi", serial=4)
events.head(10).collect()
"""),
        markdown("""## Source verification

Report field validity, duplicates, and per-machine ordering without deleting or
reordering observations. The 2026-10-18 full-file check covered 444,834 events
and matched the provider archive checksum and an independent CSV reader.
"""),
        code("verify()\n"),
    ]
    write_notebook(ROOT / "notebooks/datasets/alpi.ipynb", cells)


if __name__ == "__main__":
    main()
