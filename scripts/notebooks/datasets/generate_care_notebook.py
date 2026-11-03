#!/usr/bin/env python3
"""Generate the CARE wind-turbine SCADA waveform notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notebook_cells import ROOT, NotebookBuilder  # noqa: E402


def main() -> None:
    nb = NotebookBuilder()

    nb.markdown(
        """# CARE: Time-series waveforms

Select a wind farm and event ID to view sensor waveforms with **Matplotlib**.
This notebook includes static figures generated from real data, so they are visible in GitHub previews.

- The default selection is **Wind Farm A, event 0**. Edit the selection cell and rerun the following cells.
- Previously downloaded data is read from the configured `data_root`. No datasets are downloaded automatically.
- The full recording and the prediction detail are shown separately. Each sensor has its own vertical axis, and all rows of the selected signals are plotted.
- Yellow shading indicates the `prediction` interval; red shading indicates the provider's event interval. Timestamps are anonymized.

Select the repository's **Python (pdmdata)** kernel to run this notebook."""
    )
    nb.code(
        """from pathlib import Path
from datetime import timedelta
import sys

ROOT = next(
    path for path in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
    if (path / "pdmdata.toml").is_file() and (path / "pdmdata").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import polars as pl
from IPython.display import display
import pdmdata
from pdmdata.datasets.care.viz import plot_waveforms

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150})"""
    )
    nb.markdown(
        """## 1. Select the data

Set `WIND_FARM` to `"A"`, `"B"`, or `"C"`, and choose `EVENT_ID` from the event table below.
With `SIGNALS = None`, the first four average sensor columns are selected.
For example, for Wind Farm A you can use `SIGNALS = ["wind_speed_3_avg", "power_29_avg", "sensor_5_avg"]`.
Column names and units differ between wind farms."""
    )
    nb.code(
        """WIND_FARM = "A"
EVENT_ID = 0
SIGNALS = None
CONTEXT_DAYS = 7  # Days of context before the prediction interval

# Save under the Git-ignored data/ directory. Change this path if needed.
OUTPUT_DIR = ROOT / "data" / "figures" / "care" """[:-1]
    )
    nb.code(
        """events = pdmdata.load("care", wind_farm=WIND_FARM, table="events", lazy=False)
with pl.Config(tbl_rows=100):
    display(events.select("event_id", "asset_id", "event_label", "event_start", "event_end").sort("event_id"))"""
    )
    nb.markdown(
        """## 2. Load and inspect the recording

Raw CSV files are not modified. The `train_test` column distinguishes `train` from `prediction`;
it is not an anomaly label. Consult the event table for event labels."""
    )
    nb.code(
        """record = events.filter(pl.col("event_id") == EVENT_ID)
if record.height != 1:
    raise ValueError(f"Event {EVENT_ID} is not available in Wind Farm {WIND_FARM}. Select an event from the table above.")
event = record.row(0, named=True)
frame = pdmdata.load("care", wind_farm=WIND_FARM, event_id=EVENT_ID)
schema = frame.collect_schema()
signals = SIGNALS if SIGNALS is not None else [
    name for name, dtype in schema.items()
    if name.endswith("_avg") and dtype.is_numeric()
][:4]
print("Signals:", signals)
display(record)
display(frame.group_by("train_test").agg(
    pl.len().alias("rows"),
    pl.col("time_stamp").min().alias("start"),
    pl.col("time_stamp").max().alias("end"),
).sort("start").collect())"""
    )
    nb.code(
        """features = pdmdata.load("care", wind_farm=WIND_FARM, table="features", lazy=False)
sensor_names = [name.rsplit("_", 1)[0] if name.endswith(("_avg", "_min", "_max", "_std")) else name for name in signals]
display(features.filter(pl.col("sensor_name").is_in(sensor_names)))"""
    )
    nb.markdown(
        """## 3. Full-recording waveforms

All observations across the full recording are plotted, without truncation or downsampling.
Vertical axes show original values. Sensor scales differ, so their amplitudes should not be compared directly."""
    )
    nb.code(
        """overview = plot_waveforms(
    frame, columns=signals, event=event,
    title=f"CARE | Wind Farm {WIND_FARM} | Event {EVENT_ID} | Full recording",
)
plt.show()"""
    )
    nb.markdown(
        """## 4. Prediction detail

Show the period from `CONTEXT_DAYS` before the prediction interval begins through the end of the recording.
To inspect a different period, apply additional timestamp conditions with `frame.filter(...)` before plotting."""
    )
    nb.code(
        """prediction_start = frame.filter(pl.col("train_test") == "prediction").select(
    pl.col("time_stamp").min()
).collect().item()
if prediction_start is None:
    raise ValueError("The selected recording has no prediction interval.")
focus = frame.filter(pl.col("time_stamp") >= prediction_start - timedelta(days=CONTEXT_DAYS))
detail = plot_waveforms(
    focus, columns=signals, event=event,
    title=f"CARE | Wind Farm {WIND_FARM} | Event {EVENT_ID} | Prediction detail",
)
plt.show()"""
    )
    nb.markdown(
        """## 5. Save the figures

Save the figures as PNG files. Images are also stored in the notebook outputs for static viewing on GitHub.
Interactive Plotly figures do not work in GitHub's standard notebook preview.
[GitHub documentation](https://docs.github.com/en/repositories/working-with-files/using-files/working-with-non-code-files#working-with-jupyter-notebook-files-on-github)

CARE has known inconsistencies in Min / Max / Std statistics. This example uses Avg columns.
The provider particularly recommends Avg measurements for analyses of Wind Farm B.
[CARE source](https://zenodo.org/records/15846963)"""
    )
    nb.code(
        """OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
stem = f"care_{WIND_FARM}_{EVENT_ID}"
overview.savefig(OUTPUT_DIR / f"{stem}_overview.png", bbox_inches="tight")
detail.savefig(OUTPUT_DIR / f"{stem}_detail.png", bbox_inches="tight")
print(f"Saved {stem}_overview.png and {stem}_detail.png")"""
    )

    nb.write(ROOT / "notebooks/datasets/care.ipynb")


if __name__ == "__main__":
    main()
