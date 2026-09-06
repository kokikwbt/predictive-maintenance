# Notebooks

The notebooks are organized by purpose:

- [`datasets/`](datasets/): dataset-specific loading, inspection, and visualization
  notebooks with saved waveform examples.
- [`tasks/`](tasks/): analysis examples organized around predictive-maintenance
  tasks rather than a particular dataset.

Run notebooks with the repository root as the working directory, or keep the
default notebook working directory. Dataset notebooks must load downloaded
files from the `data_root` configured in the repository’s `pdmdata.toml`
(default: `data/raw/<dataset-id>` under the repository root).

After synchronizing the locked environment, register it as a Jupyter kernel:

```bash
uv sync --locked
uv run --locked python -m ipykernel install --user --name pdmdata --display-name "Python (pdmdata)"
uv run --locked jupyter lab
```

Select `Python (pdmdata)` as the kernel when opening a notebook.

Keep compact static waveform outputs in dataset notebooks so GitHub can display
the examples. Store README preview images in `pdmdata/<dataset-id>/assets/` and
large reports under the configured data directory. Avoid machine-specific paths
in saved tables; display filenames or relative paths. Preserve saved figures
when editing notebook sources unless the underlying example changes.
