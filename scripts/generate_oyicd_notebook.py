"""Generate the OYICD exploration notebook without downloading data."""

from pathlib import Path
import nbformat as nbf


def main():
    cells = []

    def markdown(source):
        cells.append(nbf.v4.new_markdown_cell(source))

    def code(source):
        cells.append(nbf.v4.new_code_cell(source))

    markdown('''# OYICD: operating modes and short process cycles

Explore eight process signals in 519 short recordings spanning twelve relative
months. Filename modes identify entire recordings, not annotated phases within
an operating cycle. Each capture has 2,048 samples at approximately 4 ms spacing.

[Provider and attribution](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation)
· [Dataset guide](../../pdmdata/oyicd/README.md)

Run `pdmdata.download("oyicd")` explicitly if needed. This notebook reads local
data only. The source ZIP contains two byte-identical copies of each recording;
the loader verifies agreement and counts each unique filename once.
''')
    code('''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))
import polars as pl
import matplotlib.pyplot as plt
from IPython.display import display
import pdmdata
from pdmdata.oyicd import inventory
from pdmdata.oyicd.loader import SENSOR_COLUMNS
from pdmdata.oyicd.viz import SIGNAL_LABELS, plot_waveforms, plot_coverage, plot_modes

files = inventory()
print(f"Unique recordings: {files.height}; observations: {files['samples'].sum():,}")
print(f"Local CSV copies: {files['copies'].sum()}; modes: {sorted(files['mode'].unique().to_list())}")
display(files.select("filename", "month", "mode", "samples", "copies", "span_s").head(12))
print(f"Sample spacing range: {files['min_step_s'].min():.8f} to {files['max_step_s'].max():.8f} seconds")
''')
    markdown('''## Recording coverage

Relative months are numbered 1–12; they are not calendar months. Mode coverage
is uneven, so comparing early and late recordings without controlling for mode
can mix operating differences with changes over the year.
''')
    code('''coverage = plot_coverage(files)
plt.show()
''')
    markdown('''## Select one recording

Change MONTH and MODE, then choose a filename from the matching rows. The default
uses the lexicographically first recording in month 1, mode 1. Selection does not
assume that it represents a healthy blade. Mode names are retained as numeric
IDs; no unverified mapping to machine actions is supplied.
''')
    code('''MONTH, MODE = 1, 1
candidates = files.filter((pl.col("month") == MONTH) & (pl.col("mode") == MODE)).sort("filename")
if candidates.is_empty():
    raise ValueError("No recordings match this month/mode; consult the coverage plot")
RECORDING = candidates["filename"][0]
frame = pdmdata.load("oyicd", recording=RECORDING)
print("Selected:", RECORDING)
display(pl.DataFrame({"column": SENSOR_COLUMNS, "plot_label": SIGNAL_LABELS,
    "unique_values": [frame[c].n_unique() for c in SENSOR_COLUMNS],
    "standard_deviation": [frame[c].std() for c in SENSOR_COLUMNS]}))
''')
    markdown('''## Eight-signal waveform overview

All values retain their source scale, including the large accumulated positions.
The horizontal axis subtracts only the first timestamp. No resampling, smoothing,
or concatenation is performed. Signals may cycle within a recording, but cycle
phase boundaries are not labeled.
''')
    code('''overview = plot_waveforms(frame)
assets = ROOT / "pdmdata/oyicd/assets"
assets.mkdir(parents=True, exist_ok=True)
overview.savefig(assets / "waveforms.png", dpi=100)
plt.show()
''')
    markdown('''## Zoom into the first second

The default 250 samples expose faster changes in the same recording. Adjust
START and STOP as sample indices while keeping the observed time axis.
''')
    code('''START, STOP = 0, 250
zoom = plot_waveforms(frame, start=START, stop=STOP)
plt.show()
''')
    markdown('''## One torque example per mode

Each panel uses the first available filename for that mode and shared amplitude
limits. These are independent captures, often from different relative months.
They are examples, not average mode profiles or evidence of an observed transition
from one mode to another. The selected filenames make the comparison reproducible.
''')
    code('''selected = files.sort("filename").group_by("mode", maintain_order=True).first().sort("mode")
display(selected.select("mode", "month", "filename"))
examples = [pdmdata.load("oyicd", recording=name) for name in selected["filename"]]
mode_examples = plot_modes(examples)
plt.show()
''')
    markdown('''## Implications for dependency-based segmentation

- Use the eight process signals as candidates and keep filename, month, and mode
  outside the sensor inputs. Inspect constants, trends, and channel scales first.
- The 4 ms sampling provides short-term detail, while sparse captures across
  twelve months provide a separate, much slower observation scale. The data are
  not a continuous year-long signal.
- Distinguish repeated within-capture motion from changes between filename modes.
  A constant mode label does not annotate the phases of the motion cycle.
- Accumulated position channels can have large offsets and trends; decide and
  document appropriate detrending or differencing using training data only.
- Compare dependency changes against mean/variance changes. These exploratory
  waveforms alone do not establish a multi-scale dependency model's advantage.
- Avoid splitting overlapping windows of the same capture across train/test sets.
  Also exclude duplicate source copies from evaluation.
- Blade wear, replacement times, failure events, and RUL are not explicit targets.
  Relative month should not be equated with blade age or wear severity.

Source: inIT-OWL, *One Year Industrial Component Degradation*, associated with
von Birgelen et al. (2018), *Self-Organizing Maps for Anomaly Localization and
Predictive Maintenance in Cyber-Physical Production Systems*.
Data and the derived preview are attributed under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "pdmdata", "display_name": "Python (pdmdata)", "language": "python"}
    })
    root = Path(__file__).resolve().parents[1]
    nbf.write(notebook, root / "notebooks/datasets/oyicd.ipynb")


if __name__ == "__main__":
    main()
