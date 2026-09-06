# PdMData: Datasets for Predictive Maintenance

A Python toolkit to download, load, and visualize predictive maintenance datasets
for research. Download only the datasets you need and explore them through a
common API with dataset-specific loaders.

## What is predictive maintenance?

Predictive maintenance uses equipment measurements and operating history to
assess machine condition and anticipate failures, helping inform when maintenance
is needed. Typical data include vibration, temperature, pressure, and event logs.

Machine learning, a branch of AI, can learn patterns in these data to detect
anomalies, classify faults, estimate degradation, or predict remaining useful
life (RUL). Those predictions support maintenance decisions; their usefulness
depends on the available labels, operating conditions, and evaluation design.
PdMData provides datasets and loading tools for developing and comparing these
methods.

Explore the [dataset catalog and task comparison](pdmdata/README.md) to choose
data for your experiment.

## Getting started

From the repository root, with `uv` installed:

```bash
./scripts/bootstrap.sh
uv run --locked jupyter lab
```

Setup creates the locked Python 3.11 environment and registers the
`Python (pdmdata)` notebook kernel. It does not download datasets.
Select that kernel in Jupyter and run:

```python
import pdmdata

pdmdata.summary()
pdmdata.download("cmapss")
frame = pdmdata.load("cmapss", subset="FD001", split="train")
frame.head()
```

Loaders return Polars DataFrames or LazyFrames, depending on the dataset.
Importing the package or loading data never triggers a download; download each
dataset explicitly before loading it.

To plot the downloaded example:

```python
figure = pdmdata.visualize("cmapss", "sensor_trajectory", frame, entity=1)
figure.show()
```

See the [notebooks](notebooks/README.md) for dataset exploration and task examples.

## Options

### Data location

Edit [`pdmdata.toml`](pdmdata.toml) to set the shared download and loading root:

```toml
data_root = "data/raw"
```

The default stores data inside this repository under `data/raw/<dataset-id>/`;
`data/` is ignored by Git. Relative paths resolve from the configuration file.
You can also use an absolute path or a path starting with `~`.

For a personal configuration file, set `PDMDATA_CONFIG` to its path. Changes
apply on the next download or load, and do not move existing data. Add any custom
data directory inside the repository to `.gitignore`.

### Downloads

You can also download from the command line:

```bash
uv run --locked python scripts/download.py cmapss
```

| Option | Purpose |
|---|---|
| `--variant NAME` | Select a dataset variant, such as a Backblaze quarter. |
| `--output-dir PATH` | Override the data root for this download. Set the same root in your configuration before loading. |
| `--no-extract` | Download the archive without extracting it. |
| `--overwrite` | Download again even when the archive exists. |
| `--print-command` | Display the external download command without running it. |

### Kaggle datasets and authentication

**GDD, OYICD, and PPD require the official Kaggle CLI**, included in the project
environment. Use the same download API as for other datasets:

```python
pdmdata.download("gdd")
frame = pdmdata.load("gdd", series="state")
```

PdMData first attempts the download with existing credentials or public access.
If Kaggle reports that authentication is required, an interactive terminal or
local desktop notebook starts `kaggle auth login --force`. Complete the Kaggle
sign-in and authorization in your browser; PdMData then retries the download
once. Login has a five-minute timeout. Importing or loading data never starts
login. Permission errors without an authentication indication do not trigger it.

You can authenticate in advance from a terminal:

```bash
uv run --locked kaggle auth login
```

On a remote machine without a browser, use
`uv run --locked kaggle auth login --no-launch-browser` and follow the CLI prompts.
For CI or unattended runs, configure `KAGGLE_API_TOKEN` through your environment
or secret manager; automatic login is disabled in CI and non-interactive scripts.
Remote/headless notebooks should authenticate from a terminal first.
Credentials are stored by the official Kaggle CLI, never in project metadata.
Do not commit tokens or credential files. See the
[Kaggle authentication documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication).

### Dataset selection and loading

Use `pdmdata.summary(downloadable=True)` to list datasets with automated downloads.
Loading options such as `subset`, `split`, and `wind_farm` depend on the dataset;
see its README in the [catalog](pdmdata/README.md).

Large tabular datasets may return a Polars LazyFrame. Select and filter the data
before calling `.collect()` to limit what is materialized in memory.

The [usage guide](docs/usage.md) covers manual setup, configuration overrides,
bulk downloads, and code organization.

## License

Repository code and documentation are available under the MIT License. Each
dataset remains subject to its own license and terms of use; consult its
README and source before use.
