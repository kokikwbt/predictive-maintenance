"""Generate the ALPI raw-event inspection notebook without downloading data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notebook_cells import ROOT, NotebookBuilder  # noqa: E402


def main():
    nb = NotebookBuilder()

    nb.markdown(
        """# ALPI packaging-machine alarms

Inspect original event times, alarm codes, and machine identifiers. This notebook
does not create failure labels, prediction windows, or train/test splits.

First download with `uv run --locked python scripts/download.py alpi` from the
project root. Loading below reads local files only. Source: [Mendeley Data,
version 1](https://data.mendeley.com/datasets/4nhx2x67cd/1), CC BY 4.0.
"""
    )
    nb.code(
        """import pdmdata
from pdmdata.datasets.alpi import inventory, verify

inventory()
"""
    )
    nb.markdown(
        """## Original events

Keep source order and duplicate events. Raw alarm code 0 is genuine; it is not
padding. `source_row` identifies each original CSV row after filtering. Provider
timestamps do not specify a timezone. The last alarm is not a failure endpoint.
"""
    )
    nb.code(
        """events = pdmdata.load("alpi", serial=4)
events.head(10).collect()
"""
    )
    nb.markdown(
        """## Source verification

Report field validity, duplicates, and per-machine ordering without deleting or
reordering observations. The 2026-10-18 full-file check covered 444,834 events
and matched the provider archive checksum and an independent CSV reader.
"""
    )
    nb.code("verify()\n")

    nb.write(ROOT / "notebooks/datasets/alpi.ipynb")


if __name__ == "__main__":
    main()
