# PdMData: Datasets for Predictive Maintenance

A Python toolkit to download, load, and visualize predictive-maintenance
datasets for research. Use a shared API with dataset-specific loaders; download
only what you need.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH`
- Python **3.14** (installed automatically by `uv` from `.python-version`)
- Optional: a system [libarchive](https://www.libarchive.org/) library if you
  will download **IMS** (nested ZIP/7z/RAR). macOS usually includes one;
  Debian/Ubuntu: `sudo apt-get install libarchive-dev`; see
  [Usage](docs/usage.md) for Windows notes.

Windows users can run the same `uv` commands in PowerShell, or use
Git Bash / WSL for `./scripts/bootstrap.sh`.

## Getting started

From the repository root:

```bash
./scripts/bootstrap.sh
```

Or, without the shell script:

```bash
uv sync --locked
uv run --locked python -m ipykernel install --user \
  --name pdmdata --display-name "Python (pdmdata)"
```

Confirm the install:

```bash
uv run --locked python -c "import pdmdata; pdmdata.summary()"
```

Setup creates the locked Python 3.14 environment (runtime, Jupyter, and
dev tooling) and registers the `Python (pdmdata)` notebook kernel. It does
**not** download datasets. Data lands under `data/raw/` by default
(see `pdmdata.toml`).

Start JupyterLab, select the `Python (pdmdata)` kernel, and run:

```bash
uv run --locked jupyter lab
```

```python
import pdmdata

pdmdata.summary()
pdmdata.download("cmapss")  # direct URL; no Kaggle login
frame = pdmdata.load("cmapss", subset="FD001", split="train")
frame.head()
```

Importing or loading never downloads data. Download each dataset explicitly
first. Loaders return Polars DataFrames or LazyFrames, depending on the
dataset.

```python
figure = pdmdata.visualize("cmapss", "sensor_trajectory", frame, entity=1)
figure.show()
```

## Next steps

| Goal | Where to go |
|---|---|
| Choose a dataset and compare support | [Dataset catalog](pdmdata/README.md) |
| Browse the task list and experiment views | [Task overview](pdmdata/tasks/README.md) |
| Configure paths, downloads, and Kaggle auth | [Usage guide](docs/usage.md) |
| Explore data or task examples | [Notebooks](notebooks/README.md) |
| Run tests | `uv run --locked python -m unittest discover -s tests -q` |

Open a dataset README from the catalog for selectors, schema details, and
terms of use. Open a task README for experiment bundles and scoring
protocols.

## License

Repository code and documentation are available under the MIT License. Each
dataset remains subject to its own license and terms of use; consult its
README and source before use.
