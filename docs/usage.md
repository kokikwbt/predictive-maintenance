# Usage guide

[Project overview](../README.md) · [Dataset catalog](../pdmdata/README.md)

## Installation

PdMData uses Python 3.11 and a committed `uv.lock` as its reproducible
development and test environment.

### Automated setup

From the repository root, run the bootstrap script:

```bash
./scripts/bootstrap.sh
```

This command synchronizes the locked Python 3.11 environment and registers
the `Python (pdmdata)` Jupyter kernel. It does not download any datasets.
`uv` must already be installed and available in `PATH`.

### Manual setup

```bash
uv sync --locked
uv run --locked python -m ipykernel install --user --name pdmdata --display-name "Python (pdmdata)"
```

Start JupyterLab with `uv run --locked jupyter lab`, then select `Python
(pdmdata)` as the notebook kernel.

Dataset loaders and preprocessing examples use Polars.

## Global configuration

Edit [`pdmdata.toml`](../pdmdata.toml) in the repository root to change the shared
download and loading location:

```toml
# Default: <repository>/data/raw/<dataset-id>/
data_root = "data/raw"
# Alternatively: data_root = "/Volumes/datasets/pdmdata"
# Alternatively: data_root = "~/datasets/pdmdata"
```

Relative paths are resolved from the configuration file's directory, so the
same location is used from notebooks and scripts in different directories.
The entire default `data/` directory, including archives, extracted files and
manifests, is ignored by Git. If you choose another location inside the
repository, add that directory to `.gitignore` too.

The configuration is read when downloading or loading, so edits apply to the
next operation without restarting Python. Inspect the effective location with
`pdmdata.get_settings().data_root`. Changing the setting does not move existing
files; move them yourself or download the selected dataset again.

To keep personal settings outside the tracked file, create a TOML file with
the same format and set `PDMDATA_CONFIG` to its path. An explicitly selected
missing or invalid file raises an error. In an editable checkout the repository
configuration is used by default. For a non-editable installation, the default
is `pdmdata.toml` in the working directory; without it, `data/raw` in that
directory is used. Use `PDMDATA_CONFIG` for a stable location in that case.

`download(..., output_dir=...)` and the CLI's `--output-dir` override the root
for that download only (relative to the working directory). To load from that
location later, set `data_root` to the same root in the configuration.

Neither `import pdmdata`, catalog inspection, nor `pdmdata.load(...)` downloads
data. A missing local dataset raises `FileNotFoundError`; download the desired
dataset explicitly first.

## Downloading datasets

The project supports direct URLs and Kaggle datasets:

- C-MAPSS
- GDD (Kaggle)
- GFD
- HydSys
- IMS
- MAPM (Microsoft GitHub)
- OYICD (Kaggle)
- PPD (Kaggle)

Large, opt-in downloads are also supported:

- MetroPT2 (1.2 GB CSV)
- CARE to Compare (5.5 GB ZIP)
- Backblaze Drive Stats (one selected quarter; more than 10 GB extracted)

Kaggle downloads use the official CLI. On an authentication error, PdMData
starts the official login flow in an interactive terminal or local desktop
notebook, then retries once. CI and non-interactive scripts require credentials
in advance. See [Kaggle authentication](../README.md#kaggle-datasets-and-authentication)
for browser login, remote sessions, and token-based unattended execution.

Optionally download the bulk-enabled datasets (this may still be large):

```bash
uv run --locked python scripts/download_all.py
```

Download and extract one dataset into the configured root (default: `data/raw/<dataset-id>`):

```bash
uv run --locked python scripts/download.py hydsys
```

Print the configured external command without downloading:

```bash
uv run --locked python scripts/download.py hydsys --print-command
uv run --locked python scripts/download.py gdd --print-command
```

Download a large dataset explicitly:

```bash
uv run --locked python scripts/download.py metropt2
uv run --locked python scripts/download.py care
uv run --locked python scripts/download.py backblaze --variant 2025-q1
```

Backblaze variants currently include `2025-q1`, `2025-q2`, `2025-q3`, and
`2025-q4`. These large datasets are intentionally excluded from
`download_all.py`. `bootstrap.sh` never downloads datasets.

The downloader preserves the source archive, extracts it into a separate
directory, validates expected files, and records the archive SHA-256 in
`manifest.json`. The `data/` directory is excluded from Git.

IMS contains nested ZIP, 7z, and RAR layers, which `download("ims")` extracts
automatically. The locked Python environment includes `libarchive-c`; its
native `libarchive` library must also be available. macOS includes a system
copy. If it is missing or too old, use `brew install libarchive`; on
Debian/Ubuntu use `sudo apt-get install libarchive-tools`. Windows requires a
compatible libarchive DLL. Set `LIBARCHIVE` to the shared-library path when it
cannot be discovered automatically. See the [IMS guide](../pdmdata/ims/README.md)
for recording selectors and source-format details.

## Usage

Import the package from the repository root:

```python
import pdmdata

# Display the dataset catalog
pdmdata.summary()

# Download a supported dataset into data/raw
pdmdata.download("hydsys")

# Load downloaded source data as a Polars DataFrame
frame = pdmdata.load("hydsys", sensor="PS1", cycle=0)
```

The large tabular datasets use lazy Polars scans by default:

```python
metro = pdmdata.load("metropt2")  # pl.LazyFrame
care_event = pdmdata.load("care", wind_farm="A", event_id=0)
drives = pdmdata.load("backblaze", variant="2025-q1")
```

Use `.select(...)`, `.filter(...)`, and then `.collect()` so that Polars reads
only the required data. CARE loads one recording at a time because the archive
contains many separate event and normal-operation datasets.

Create a registered interactive Plotly figure:

```python
frame = pdmdata.load("cmapss", subset="FD001", split="train")
figure = pdmdata.visualize(
    "cmapss",
    "sensor_trajectory",
    frame,
    entity=1,
)
figure.show()
```

See the [visualization showcase](../notebooks/datasets/visualization-showcase.ipynb)
for time-series, distribution, and event-timeline examples.

Show only datasets supported by an automated download method:

```python
pdmdata.summary(downloadable=True)
```

Dataset files are not stored under `pdmdata/`. That directory contains
metadata and analysis-library code; downloaded source data belongs under the
configured `data_root/<dataset-id>` (default: `data/raw/<dataset-id>`). Dataset
loaders and notebooks use the same setting.

## Code organization

Keep dataset-specific code beside its metadata and documentation:

```text
pdmdata/
  config.py              # Shared user settings
  loaders.py             # Common load() entry point and dispatch registry
  io.py                  # Reusable file discovery and basic readers
  download.py            # Shared download/extraction/integrity machinery
  care/
    __init__.py          # Direct CARE API
    loader.py            # CARE layout, selectors, parsing and normalization
    validation.py        # CARE-specific verification
    viz.py               # CARE-specific plots
    metadata.json        # Sources, expected files and dataset metadata
    README.md
  cmapss/
    loader.py            # C-MAPSS-specific columns and loading
    ...
```

Every dataset's loading logic belongs in its own `loader.py`. Common helpers
stay in `io.py`; the global loader registry only routes calls. Put future
preprocessing or validation specific to one dataset in that dataset's directory,
for example `care/preprocessing.py`. CLI scripts should call that implementation.
To add a dataset loader, implement `load()` in its directory and register its
module in `pdmdata/loaders.py`.

For example, these call the same CARE implementation:

```python
import pdmdata
from pdmdata.care import load as load_care

frame = pdmdata.load("care", wind_farm="A", event_id=0)
train = load_care(wind_farm="A", event_id=0, split="train")
```

Loading interprets the source layout and applies explicit selections in memory;
it does not change the downloaded CSVs or download missing data.

## Analysis conventions

### Run-to-failure data

Loaders retain dataset-specific schemas rather than adding a universal event or
censoring column. For modeling, identify the trajectory and time columns, select
sensor features, and derive endpoints or censoring indicators using that dataset's
documentation. Keep derived targets separate from model inputs.

## Notebooks

Jupyter notebooks are organized by purpose:

- [`notebooks/datasets/`](../notebooks/datasets/) contains loading,
  inspection, and visualization notebooks.
- [`notebooks/tasks/`](../notebooks/tasks/) contains predictive-maintenance
  analysis examples that can apply across datasets.

See the [notebook guide](../notebooks/README.md) for the organization and
contribution guidelines.


## Recommended projects

- [IBM FailureSensorIQ](https://github.com/IBM/FailureSensorIQ): a dataset and
  benchmark for evaluating language-model reasoning about relationships
  between industrial sensor behavior and failure modes.
- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench): a framework and
  benchmark for building, orchestrating, and evaluating AI agents for
  industrial asset operations and maintenance.

## References

### Predictive-maintenance background

- [Predictive maintenance overview](https://en.wikipedia.org/wiki/Predictive_maintenance)
- [Azure architecture guide for predictive-maintenance solutions](https://learn.microsoft.com/azure/architecture/data-science-process/predictive-maintenance-playbook)
- [Types of proactive maintenance](https://solutions.borderstates.com/types-of-proactive-maintenance/)
- [Kaggle community guide to common dataset licenses](https://www.kaggle.com/general/116302)
