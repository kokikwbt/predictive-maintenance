# Scripts

Command-line tools for environment bootstrap, dataset downloads, catalog
documentation, notebook generation, and dataset-specific maintenance.

[Project overview](../README.md) · [Usage guide](../docs/usage.md) · [Dataset catalog](../pdmdata/README.md)

## Directory layout

```text
scripts/
  bootstrap.sh                  # Sync locked uv environment and Jupyter kernel
  download.py                   # Download and validate one dataset
  download_all.py               # Download bulk-enabled datasets
  generate_docs.py              # Regenerate dataset READMEs and catalog tables
  notebooks/                    # Notebook generators and shared cell builder
    notebook_cells.py           # Shared notebook JSON builder and writer
    generate_all.py             # Regenerate all script-managed notebooks
    generate_task_notebooks.py  # Generate task notebooks under notebooks/tasks/
    generate_visualization_showcase.py
    datasets/                   # Per-dataset notebook generators
      generate_<id>_notebook.py
  maintenance/                  # Offline dataset verification and sample plots
    verify_care.py              # Full offline verification for CARE
    generate_backblaze_samples.py
```

## User commands

Run all commands from the repository root.

### Bootstrap the environment

Synchronize the locked Python 3.14 environment and register the
`Python (pdmdata)` Jupyter kernel (does not download datasets):

```bash
./scripts/bootstrap.sh
```

### Download datasets

Download and extract one dataset into the configured `data_root`
(default: `data/raw/<dataset-id>`):

```bash
uv run --locked python scripts/download.py cmapss
uv run --locked python scripts/download.py backblaze --variant 2025-q1
```

Inspect the external download command without executing it:

```bash
uv run --locked python scripts/download.py hydsys --print-command
```

Download all bulk-enabled datasets (excludes large opt-in datasets):

```bash
uv run --locked python scripts/download_all.py
```

## Maintainer commands

### Regenerate catalog and dataset documentation

Synchronize `pdmdata/README.md` and the generated metadata blocks in
`pdmdata/datasets/<id>/README.md` from `metadata.json`:

```bash
uv run --locked python scripts/generate_docs.py
```

### Regenerate notebooks

Regenerate all script-managed dataset, showcase, and task notebooks
(saved cell outputs on unchanged code cells are preserved automatically):

```bash
uv run --locked python scripts/notebooks/generate_all.py
```

Or regenerate a specific group or dataset notebook:

```bash
uv run --locked python scripts/notebooks/generate_task_notebooks.py
uv run --locked python scripts/notebooks/generate_visualization_showcase.py
uv run --locked python scripts/notebooks/datasets/generate_cmapss_notebook.py
```

### Dataset maintenance utilities

Verify all local CARE files and write an offline inventory report:

```bash
uv run --locked python scripts/maintenance/verify_care.py
```

Regenerate compact Backblaze preview plots from a local `2025-q1` archive:

```bash
uv run --locked python scripts/maintenance/generate_backblaze_samples.py
```
