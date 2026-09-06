"""Generate C-MAPSS trajectory, operating-setting, and RUL examples."""

from pathlib import Path
import nbformat as nbf


def main():
    cells = []
    def markdown(source):
        cells.append(nbf.v4.new_markdown_cell(source))
    def code(source):
        cells.append(nbf.v4.new_code_cell(source))

    markdown('''# C-MAPSS: engine degradation and operating-condition changes

Explore all four subsets of NASA's simulated engine benchmark. Each row is
one observation per operating cycle, with 21 sensor values and three operating
settings. These are cycle-level trajectories, not raw high-frequency vibration.

[NASA source](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
· [Dataset guide](../../pdmdata/cmapss/README.md)

Run `pdmdata.download("cmapss")` explicitly if needed. This notebook reads local
data and saves compact Matplotlib figures for GitHub. Unit IDs start at 1 and
are local to each subset and split; train unit 1 and test unit 1 are different
engines.
''')
    code('''from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))
import polars as pl
import matplotlib.pyplot as plt
from IPython.display import display
import pdmdata
from pdmdata.cmapss import inventory, load, rul
from pdmdata.cmapss.loader import SENSOR_COLUMNS
from pdmdata.cmapss.viz import plot_waveforms, plot_operating_settings, plot_rul

summary = inventory()
display(summary)
print(f"Train/test trajectories: {summary['units'].sum():,}; cycle rows: {summary['rows'].sum():,}")
''')
    markdown('''## Subsets and source inventory

FD001/FD003 use one operating condition; FD002/FD004 use six. FD001/FD002
simulate HPC degradation; FD003/FD004 include HPC and fan degradation modes.
The files do not provide per-cycle fault-mode or fault-onset labels.

The bundled readme reverses the FD004 train/test counts: actual files contain
249 train and 248 test engines, with 248 test-end RUL targets. Inventory and
loading use the actual files. There are 709 train and 707 test trajectories.
''')
    markdown('''## FD001: six sensor trajectories from one engine

Use `unit=1` to select one engine. Its observed training trajectory ends at
failure. The selected sensors retain their original values; no smoothing,
normalization, clipping, or healthy/faulty threshold is applied.
''')
    code('''SUBSET = "FD001"
UNIT = 1
train_unit = load(SUBSET, "train", unit=UNIT, with_rul=True)
display(train_unit.select("unit_number", "cycle", "sensor_2", "sensor_11", "RUL").head())
waveforms = plot_waveforms(train_unit, subset=SUBSET)
assets = ROOT / "pdmdata/cmapss/assets"
assets.mkdir(parents=True, exist_ok=True)
waveforms.savefig(assets / "waveforms.png", dpi=100)
plt.show()
''')
    markdown('''## Feature variability in the training split

Use the 21 sensors as candidate graph variables and keep operating settings as
separate context when that fits the model. Some sensors are constant or nearly
constant in a subset. Inspect variability using **training data only** before
choosing columns and fitting a scaler. Dropping exact constants alone does not
identify every numerically uninformative channel.
''')
    code('''training = load(SUBSET, "train")
variability = pl.DataFrame({
    "sensor": SENSOR_COLUMNS,
    "unique_values": [training[c].n_unique() for c in SENSOR_COLUMNS],
    "standard_deviation": [training[c].std() for c in SENSOR_COLUMNS],
})
display(variability)
nonconstant = variability.filter(pl.col("unique_values") > 1)["sensor"].to_list()
X = train_unit.select(nonconstant)
print("Exact nonconstant sensor columns from FD001 training:", len(nonconstant))
print("Selected unit sensor-only shape:", X.shape)
''')
    markdown('''## FD002: multiple operating conditions change sensor values

The same six sensors can show much larger between-cycle jumps under changing
operating settings. The next figures use the first training engine of FD002,
first over its complete trajectory and then over its first 80 cycles. Do not
interpret every jump as a degradation event.
''')
    code('''multiple_conditions = load("FD002", "train", unit=1)
condition_waveforms = plot_waveforms(multiple_conditions, subset="FD002")
plt.show()
settings = plot_operating_settings(multiple_conditions, subset="FD002", stop_cycle=80)
settings.savefig(assets / "operating_settings.png", dpi=100)
plt.show()
''')
    markdown('''## Explicit RUL: train failure endpoints and official test offsets

`with_rul=True` adds uncapped RUL, measured in remaining operating cycles:

- Train: `max_observed_cycle - cycle`, ending at 0.
- Test: `max_observed_cycle - cycle + official_test_end_RUL`, ending at the
  supplied offset rather than 0.

`rul(subset)` returns the official offsets with explicit unit IDs. The legacy
`load(subset, split="rul")` API keeps its single RUL column in source row order.
Test targets are evaluation information: do not use them to fit the encoder,
select features, normalize inputs, or set segmentation thresholds.
''')
    code('''test_unit = load(SUBSET, "test", unit=UNIT, with_rul=True)
display(rul(SUBSET).head())
print("Train final RUL:", train_unit["RUL"][-1])
print("Test final RUL:", test_unit["RUL"][-1])
rul_figure = plot_rul(train_unit, test_unit, subset=SUBSET)
plt.show()
''')
    markdown('''## Research implications

- Cycle is a discrete observation index; the files do not define an elapsed
  duration in seconds or a within-cycle waveform. Do not assign an Hz rate.
- Start with FD001 for degradation under one operating condition. FD002/FD004
  introduce operating-setting changes that affect sensor dependence as well as
  marginal signal levels. Separate operating-condition effects from degradation.
- The three recorded settings are observed covariates; an operating-state ID
  produced by clustering them would be a derived label, not supplied ground truth.
- There are no per-cycle fault-onset or segmentation labels. RUL supervision
  supports prognosis but is not a direct segmentation annotation.
- Uncapped RUL decreases from the start even during healthy operation; this
  target definition does not assume that damage began at the first observation.
- Keep engines separate and split validation by engine. Never place windows
  from one engine in both training and validation. Unit identity is the tuple
  `(subset, split, unit_number)`.
- `load()` retains all 26 source columns by default. Feature selection, scaling,
  interpolation, and RUL caps are modeling choices and are not applied silently.
- `verify()` checks complete engine/row counts, 26-column finite matrices,
  consecutive cycles, and one valid RUL offset per test engine. The ignored
  raw-data directory contains the verification report from preparation.

Citation: A. Saxena and K. Goebel (2008). *Turbofan Engine Degradation Simulation
Data Set*, NASA Prognostics Data Repository. Source-format details are from the
bundled readme and *Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation* (Saxena et al., 2008). Figures select source trajectories
and add axes; values are unchanged.
''')
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "pdmdata", "display_name": "Python (pdmdata)", "language": "python"}
    })
    nbf.write(notebook, Path(__file__).resolve().parents[1] / "notebooks/datasets/cmapss.ipynb")


if __name__ == "__main__":
    main()
