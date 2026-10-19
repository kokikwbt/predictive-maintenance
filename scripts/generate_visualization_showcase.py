#!/usr/bin/env python3
"""Generate the interactive dataset-visualization showcase notebook."""

from __future__ import annotations

from notebook_cells import ROOT, code, markdown, write_notebook

OUTPUT = ROOT / "notebooks" / "datasets" / "visualization-showcase.ipynb"


CELLS: list[dict[str, object]] = [
    markdown(
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
        identifier="introduction",
    ),
    code(
        """from pathlib import Path
import sys

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))

import polars as pl
import pdmdata

pdmdata.available_plots()""",
        identifier="setup",
    ),
    markdown(
        """## C-MAPSS sensor trajectory

The reusable time-series function filters one engine, applies the configured
point limit, and uses C-MAPSS-specific sensor labels.""",
        identifier="cmapss-heading",
    ),
    code(
        """pdmdata.download("cmapss")
cmapss = pdmdata.load("cmapss", subset="FD001", split="train")
cmapss_figure = pdmdata.visualize(
    "cmapss",
    "sensor_trajectory",
    cmapss,
    entity=1,
)
cmapss_figure.show()""",
        identifier="cmapss-plot",
    ),
    markdown(
        """## GFD condition comparison

The same plotting layer renders a grouped distribution when the dataset
configuration describes a feature and condition column.""",
        identifier="gfd-heading",
    ),
    code(
        """pdmdata.download("gfd")
gfd = pl.concat([
    pdmdata.load("gfd", condition="healthy", load=50),
    pdmdata.load("gfd", condition="broken", load=50),
])
gfd_figure = pdmdata.visualize("gfd", "condition_distribution", gfd)
gfd_figure.show()""",
        identifier="gfd-plot",
    ),
    markdown(
        """## MAPM telemetry and errors

MAPM demonstrates two specifications for one dataset. Telemetry is rendered as
multiple lines, while error records use an event timeline.""",
        identifier="mapm-heading",
    ),
    code(
        """pdmdata.download("mapm")
telemetry = pdmdata.load("mapm", table="telemetry")
telemetry_figure = pdmdata.visualize(
    "mapm",
    "telemetry",
    telemetry,
    entity=1,
)
telemetry_figure.show()""",
        identifier="mapm-telemetry",
    ),
    code(
        """errors = pdmdata.load("mapm", table="errors")
error_figure = pdmdata.visualize(
    "mapm",
    "error_timeline",
    errors,
    entity=1,
)
error_figure.show()""",
        identifier="mapm-events",
    ),
    markdown(
        """## Large, opt-in datasets

The following examples do not start downloads automatically. Download only the
dataset you need with `scripts/download.py` before running its cell. The
loaders return Polars `LazyFrame` objects so aggregation and bounded projection
happen before Plotly receives data.""",
        identifier="large-heading",
    ),
    markdown(
        """### MetroPT2 compressor signals

Run `uv run --locked python scripts/download.py metropt2` once before this example.""",
        identifier="metropt2-heading",
    ),
    code(
        """metropt2 = pdmdata.load("metropt2")
metropt2_figure = pdmdata.visualize(
    "metropt2",
    "sensor_signals",
    metropt2,
)
metropt2_figure.show()""",
        identifier="metropt2-plot",
    ),
    markdown(
        """### CARE wind-turbine SCADA signals

Run `uv run --locked python scripts/download.py care` once before this example.
Select wind farm A and event 0 through the CARE loader. Other farms and events
can be selected with `wind_farm` and `event_id`.""",
        identifier="care-heading",
    ),
    code(
        """care = pdmdata.load("care", wind_farm="A", event_id=0)
care_figure = pdmdata.visualize(
    "care",
    "scada_signals",
    care,
)
care_figure.show()""",
        identifier="care-plot",
    ),
    markdown(
        """### Backblaze daily failure rate

Run `uv run --locked python scripts/download.py backblaze --variant 2025-q1` once before this
example. Daily aggregation is executed lazily over the quarter.""",
        identifier="backblaze-heading",
    ),
    code(
        """backblaze = pdmdata.load("backblaze", variant="2025-q1")
backblaze_figure = pdmdata.visualize(
    "backblaze",
    "daily_failure_rate",
    backblaze,
)
backblaze_figure.show()""",
        identifier="backblaze-plot",
    ),
    markdown(
        """## Extending the showcase

Add typed specifications to `pdmdata/<dataset-id>/viz.py` and
register them in `pdmdata/visualization/registry.py`. Extend the shared
plotting functions only when a genuinely reusable chart type is needed.""",
        identifier="extension",
    ),
]


def main() -> None:
    write_notebook(OUTPUT, CELLS)


if __name__ == "__main__":
    main()
