#!/usr/bin/env python3
"""Generate the interactive dataset-visualization showcase notebook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "datasets" / "visualization-showcase.ipynb"


def markdown(identifier: str, source: str) -> Dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": identifier,
        "metadata": {},
        "source": source.splitlines(True),
    }


def code(identifier: str, source: str) -> Dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": identifier,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


CELLS: List[Dict[str, object]] = [
    markdown(
        "introduction",
        """# Interactive visualization showcase

This notebook demonstrates the reusable Plotly visualization API with six
predictive-maintenance data layouts:

- C-MAPSS run-to-failure sensor trajectories;
- GFD healthy and broken-tooth vibration distributions;
- MAPM telemetry and event timelines.
- MetroPT2 compressor signals;
- CARE wind-turbine SCADA signals;
- Backblaze fleet failure rates.

Data preparation remains in Polars. Dataset-specific modules supply column
names, labels, and defaults, while the shared API returns Plotly figures.""",
    ),
    code(
        "setup",
        """from pathlib import Path
import sys

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))

import polars as pl
import pdmdata

pdmdata.available_plots()""",
    ),
    markdown(
        "cmapss-heading",
        """## C-MAPSS sensor trajectory

The reusable time-series function filters one engine, applies the configured
point limit, and uses C-MAPSS-specific sensor labels.""",
    ),
    code(
        "cmapss-plot",
        """pdmdata.download("cmapss")
cmapss = pdmdata.load("cmapss", subset="FD001", split="train")
cmapss_figure = pdmdata.visualize(
    "cmapss",
    "sensor_trajectory",
    cmapss,
    entity=1,
)
cmapss_figure.show()""",
    ),
    markdown(
        "gfd-heading",
        """## GFD condition comparison

The same plotting layer renders a grouped distribution when the dataset
configuration describes a feature and condition column.""",
    ),
    code(
        "gfd-plot",
        """pdmdata.download("gfd")
gfd = pl.concat([
    pdmdata.load("gfd", condition="healthy", load=50),
    pdmdata.load("gfd", condition="broken", load=50),
])
gfd_figure = pdmdata.visualize("gfd", "condition_distribution", gfd)
gfd_figure.show()""",
    ),
    markdown(
        "mapm-heading",
        """## MAPM telemetry and errors

MAPM demonstrates two specifications for one dataset. Telemetry is rendered as
multiple lines, while error records use an event timeline.""",
    ),
    code(
        "mapm-telemetry",
        """pdmdata.download("mapm")
telemetry = pdmdata.load("mapm", table="telemetry")
telemetry_figure = pdmdata.visualize(
    "mapm",
    "telemetry",
    telemetry,
    entity=1,
)
telemetry_figure.show()""",
    ),
    code(
        "mapm-events",
        """errors = pdmdata.load("mapm", table="errors")
error_figure = pdmdata.visualize(
    "mapm",
    "error_timeline",
    errors,
    entity=1,
)
error_figure.show()""",
    ),
    markdown(
        "large-heading",
        """## Large, opt-in datasets

The following examples do not start downloads automatically. Download only the
dataset you need with `scripts/download.py` before running its cell. The
loaders return Polars `LazyFrame` objects so aggregation and bounded projection
happen before Plotly receives data.""",
    ),
    markdown(
        "metropt2-heading",
        """### MetroPT2 compressor signals

Run `uv run --locked python scripts/download.py metropt2` once before this example.""",
    ),
    code(
        "metropt2-plot",
        """metropt2 = pdmdata.load("metropt2")
metropt2_figure = pdmdata.visualize(
    "metropt2",
    "sensor_signals",
    metropt2,
)
metropt2_figure.show()""",
    ),
    markdown(
        "care-heading",
        """### CARE wind-turbine SCADA signals

Run `uv run --locked python scripts/download.py care` once before this example.
Select wind farm A and event 0 through the CARE loader. Other farms and events
can be selected with `wind_farm` and `event_id`.""",
    ),
    code(
        "care-plot",
        """care = pdmdata.load("care", wind_farm="A", event_id=0)
care_figure = pdmdata.visualize(
    "care",
    "scada_signals",
    care,
)
care_figure.show()""",
    ),
    markdown(
        "backblaze-heading",
        """### Backblaze daily failure rate

Run `uv run --locked python scripts/download.py backblaze --variant 2025-q1` once before this
example. Daily aggregation is executed lazily over the quarter.""",
    ),
    code(
        "backblaze-plot",
        """backblaze = pdmdata.load("backblaze", variant="2025-q1")
backblaze_figure = pdmdata.visualize(
    "backblaze",
    "daily_failure_rate",
    backblaze,
)
backblaze_figure.show()""",
    ),
    markdown(
        "extension",
        """## Extending the showcase

Add typed specifications to `pdmdata/<dataset-id>/viz.py` and
register them in `pdmdata/visualization/registry.py`. Extend the shared
plotting functions only when a genuinely reusable chart type is needed.""",
    ),
]


def main() -> None:
    document = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (pdmdata)",
                "language": "python",
                "name": "pdmdata",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
