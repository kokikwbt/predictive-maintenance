# Usage guide

[Project overview](../README.md) · [Dataset catalog](../pdmdata/README.md)

## Installation

PdMData targets **Python 3.14** and uses a committed `uv.lock` as its
reproducible development and test environment. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/) first; `uv`
will fetch the matching CPython build from `.python-version`.

### Automated setup

From the repository root, run the bootstrap script:

```bash
./scripts/bootstrap.sh
```

This command synchronizes the locked Python 3.14 environment and registers
the `Python (pdmdata)` Jupyter kernel. It does not download any datasets.
On Windows without Bash, use the manual steps below in PowerShell.

Confirm:

```bash
uv run --locked python -c "import pdmdata; pdmdata.summary()"
```

### Manual setup

```bash
uv sync --locked
uv run --locked python -m ipykernel install --user --name pdmdata --display-name "Python (pdmdata)"
```

`uv sync` installs the package dependencies plus the default `dev` and
`jupyter` groups from `pyproject.toml`. For a minimal runtime-only
environment (no Jupyter, no lint tools), use
`uv sync --locked --no-default-groups`. Install Jupyter alone with
`uv sync --locked --no-default-groups --group jupyter`.

This repository is developed with `uv`. A pip editable install
(`pip install -e ".[jupyter]"`) can work for exploration, but the locked
`uv` environment is the supported path for tests and notebooks.

Start JupyterLab with `uv run --locked jupyter lab`, then select `Python
(pdmdata)` as the notebook kernel.

Dataset loaders and preprocessing examples use Polars.

### Tests and lint

From the repository root, after `uv sync --locked`:

```bash
uv run --locked python -m unittest discover -s tests -p 'test_*.py' -q
uv run --locked ruff check pdmdata tests scripts
```

GitHub Actions runs the same checks on each push and pull request.

### Troubleshooting

| Symptom | What to check |
|---|---|
| `uv: command not found` | Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and reopen the terminal. |
| Wrong Python version | Run `uv python install` from the repo root so `.python-version` (3.14) is used. |
| Kernel `Python (pdmdata)` missing | Re-run the `ipykernel install` command above, then restart Jupyter. |
| `FileNotFoundError` on `load` | Download first: `pdmdata.download("...")` or `scripts/download.py`. |
| Kaggle auth prompts | See [Kaggle authentication](#kaggle-authentication) below. |
| IMS extract fails | Install a system libarchive and set `LIBARCHIVE` if needed (see IMS notes below). |

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

Use `pdmdata.download("<id>")` or
`uv run --locked python scripts/download.py <id>`. Files go under the
configured `data_root` (default `data/raw/<id>/`). This project does not ship
or redistribute dataset files; downloads come from the original providers (or
their documented mirrors). Review each dataset’s license and terms before use
and proceed at your own responsibility (see the root README).

**Direct URL / no account (good first choices):** ALPI, C-MAPSS, GFD, HydSys,
IMS, MAPM, MetroPT2, CARE, Backblaze (pick a quarter), SCANIA Component X,
XJTU-SY.

**Kaggle account required:** GDD, OYICD, PPD, and the N-CMAPSS Kaggle mirror
(`ncmapss`, one HDF5 variant at a time).

**Large / opt-in (excluded from `download_all.py`):** MetroPT2 (~1.2 GB),
CARE (~5.5 GB ZIP), Backblaze (one quarter; >10 GB extracted), SCANIA
Component X (~1.65 GB download / ~3.3 GB extracted), N-CMAPSS subsets,
XJTU-SY (~5.4 GB ZIP). ALPI is also excluded from bulk download.

Compare sizes, licenses, and selectors in the
[dataset catalog](../pdmdata/README.md).

### Kaggle authentication

**GDD, OYICD, PPD, and N-CMAPSS (Kaggle mirror) use the official Kaggle CLI**,
included in the project environment. Example:

```python
pdmdata.download("gdd")
frame = pdmdata.load("gdd", series="state")
```

PdMData first attempts the download with existing credentials or public
access. If Kaggle reports that authentication is required, an interactive
terminal or local desktop notebook starts `kaggle auth login --force`.
Complete the Kaggle sign-in and authorization in your browser; PdMData then
retries the download once. Login has a five-minute timeout. Importing or
loading data never starts login. Permission errors without an authentication
indication do not trigger it.

You can authenticate in advance from a terminal:

```bash
uv run --locked kaggle auth login
```

On a remote machine without a browser, use
`uv run --locked kaggle auth login --no-launch-browser` and follow the CLI
prompts. For CI or unattended runs, configure `KAGGLE_API_TOKEN` through
your environment or secret manager; automatic login is disabled in CI and
non-interactive scripts. Remote/headless notebooks should authenticate from
a terminal first. Credentials are stored by the official Kaggle CLI, never
in project metadata. Do not commit tokens or credential files. See the
[Kaggle authentication documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).

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
uv run --locked python scripts/download.py scania_x
```

| Option | Purpose |
|---|---|
| `--variant NAME` | Select a dataset variant, such as a Backblaze quarter. |
| `--output-dir PATH` | Override the data root for this download. Set the same root in your configuration before loading. |
| `--no-extract` | Download the archive without extracting it. |
| `--overwrite` | Download again even when the archive exists. |
| `--print-command` | Display the external download command without running it. |

Backblaze variants currently include `2025-q1`, `2025-q2`, `2025-q3`, and
`2025-q4`. N-CMAPSS downloads one HDF5 subset at a time (default `ds01`);
the official NASA nested ZIP is about 15 GB. XJTU-SY is a single ~5.4 GB
ZIP mirror. SCANIA Component X needs about 3.3 GB after extraction. These
large datasets are intentionally excluded from `download_all.py`.
`bootstrap.sh` never downloads datasets.

The downloader preserves the source archive, extracts it into a separate
directory, validates expected files, and records the archive SHA-256 in
`manifest.json`. Multi-file sources are bundled locally into one ZIP; when
the provider publishes per-file checksums, each file is verified before
bundling. The `data/` directory is excluded from Git.

IMS contains nested ZIP, 7z, and RAR layers, which `download("ims")` extracts
automatically. The locked Python environment includes `libarchive-c`; its
native `libarchive` library must also be available. macOS includes a system
copy. If it is missing or too old, use `brew install libarchive`; on
Debian/Ubuntu use `sudo apt-get install libarchive-tools`. Windows requires a
compatible libarchive DLL. Set `LIBARCHIVE` to the shared-library path when it
cannot be discovered automatically. See the [IMS guide](../pdmdata/datasets/ims/README.md)
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

Shared concerns and dataset adapters are separated:

```text
pdmdata/
  config.py                 # Shared user settings
  catalog/                  # Metadata discovery and doc generation
  datasets/
    load.py                 # Common load() dispatch registry
    care/
      __init__.py           # Direct CARE API
      loader.py             # CARE layout, selectors, parsing
      validation.py         # CARE-specific verification
      viz.py                # CARE-specific plots
      metadata.json
      README.md
    cmapss/
      loader.py
      ...
  io/                       # Download, path discovery, Polars readers
  tasks/
    README.md               # Task package index
    taxonomy.py             # Canonical task ids
    rul/                    # RUL package (prepare.py, README)
    tte/                    # TTE package (prepare.py, README)
  visualization/            # Shared plotting models and registry
```

Every dataset's loading logic belongs in its own
`datasets/<id>/loader.py`. Common helpers stay in `pdmdata.io`; the global
loader registry only routes calls. Put future preprocessing or validation
specific to one dataset in that dataset's directory, for example
`datasets/care/preprocessing.py`. CLI scripts should call that implementation.
To add a dataset loader, implement `load()` in its directory and register its
module in `pdmdata/datasets/load.py`.

For example, these call the same CARE implementation:

```python
import pdmdata
from pdmdata.datasets.care import load as load_care

frame = pdmdata.load("care", wind_farm="A", event_id=0)
train = load_care(wind_farm="A", event_id=0, split="train")
```

Loading interprets the source layout and applies explicit selections in memory;
it does not change the downloaded CSVs or download missing data.

## RUL task view (C-MAPSS)

Dataset loaders stay source-faithful. For remaining-useful-life experiments,
build a Polars-tabular bundle in the task layer:

```python
from pdmdata.tasks.rul import eval_cmapss, prepare_cmapss

pdmdata.download("cmapss")
bundle = prepare_cmapss(
    "FD001",
    rul_cap=125,
    validation_fraction=0.2,
    drop_constant_features=True,
)
X, y, groups = bundle.train.to_numpy()
# Fit an estimator on X, y (keep groups for unit-aware CV if needed).
eval_split = bundle.test_eval_split()
predictions = model.predict(eval_split.X().to_numpy())
eval_cmapss(bundle, predictions)
```

RUL is treated as a time-to-event profile (event = end of useful life). Train
rows include `RUL`, `time_to_event`, and `event_observed`. Test feature rows
omit targets; score the official one-row-per-engine protocol with
`test_eval_split()` / `eval_cmapss`. See the
[RUL task guide](../pdmdata/tasks/rul/README.md) and the
[RUL notebook](../notebooks/tasks/remaining-useful-life-prediction.ipynb).

The same package also exposes `prepare_ncmapss` and `prepare_xjtu_sy` for
N-CMAPSS and XJTU-SY. XJTU-SY scores every held-out snapshot rather than only
the final cycle.

## TTE task view (C-MAPSS)

Time-to-event uses the same C-MAPSS trajectories with `time_to_event` as the
primary target and explicit `event_observed` flags:

```python
from pdmdata.tasks.tte import prepare_cmapss, eval_cmapss

bundle = prepare_cmapss(
    "FD001",
    time_cap=125,
    validation_fraction=0.2,
    drop_constant_features=True,
)
X, y, groups = bundle.train.to_numpy()
entities = bundle.entity_table()  # failures + right-censored test engines
eval_cmapss(bundle, predictions)
```

`entity_table()` keeps NASA test engines censored at the last observed cycle
(no official RUL offset). Official remaining-time scoring stays on
`test_eval_split()`. See the [TTE task guide](../pdmdata/tasks/tte/README.md)
and the [TTE notebook](../notebooks/tasks/time-to-event-prediction.ipynb).

Matching TTE prepare helpers are available as `prepare_ncmapss` and
`prepare_xjtu_sy` in `pdmdata.tasks.tte`.

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
- [Types of proactive maintenance](https://solutions.borderstates.com/blog/types-of-proactive-maintenance/)
- [Kaggle community guide to common dataset licenses](https://www.kaggle.com/general/116302)


### Optional bounded parallel HTTP downloads

For large public files on servers advertising byte-range support, set
`PDMDATA_DOWNLOAD_WORKERS=8` (allowed range: 1 to 8, default: 1).
The downloader validates each response range and retries an interrupted chunk
up to three times, resuming its received bytes. It publishes the temporary file
only after all chunks finish; source checksums remain mandatory when provided.
Servers without range support and small files retain the ordinary download path.
This option changes transport only, not dataset parsing or file contents.
