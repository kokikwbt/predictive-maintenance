# Notebooks

The notebooks are organized by purpose:

- [`datasets/`](datasets/): dataset-specific loading, inspection, and visualization
  notebooks with saved waveform examples.
- [`tasks/`](tasks/): analysis examples organized around predictive-maintenance
  tasks rather than a particular dataset.

## Setup

From the repository root (after [installing with uv](../README.md#getting-started)):

```bash
uv sync --locked
uv run --locked python -m ipykernel install --user \
  --name pdmdata --display-name "Python (pdmdata)"
uv run --locked jupyter lab
```

Select `Python (pdmdata)` as the kernel when opening a notebook.

Run notebooks with the repository root as the working directory, or keep the
default notebook working directory. Dataset notebooks load files from the
`data_root` in `pdmdata.toml` (default: `data/raw/<dataset-id>` under the
repository root). Download a dataset before running its notebook, for example
`uv run --locked python scripts/download.py cmapss`.

## Authoring notes

Keep compact static waveform outputs in dataset notebooks so GitHub can display
the examples. Store README preview images in
`pdmdata/datasets/<dataset-id>/assets/` and large reports under the configured
data directory. Avoid machine-specific paths in saved tables; display filenames
or relative paths. Preserve saved figures when editing notebook sources unless
the underlying example changes.
