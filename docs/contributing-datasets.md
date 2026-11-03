# Contributing dataset adapters

This guide defines the acceptance criteria, standard directory layout, and
implementation checklist for adding or updating a dataset adapter in PdMData.

[Documentation index](README.md) · [Dataset catalog](../pdmdata/README.md) · [Dataset candidates](dataset-candidates.md)

## Adapter acceptance criteria

1. Pin a provider version, file manifest, download URLs, checksums when supplied,
   citation, and dataset-specific terms. Retain the source terms; do not infer
   a data license from a mirror or a code repository alone.
2. Inspect source files through a source-faithful adapter. Establish entity
   identifiers, physical units, timestamps, ordering, duplicates, missingness,
   event meanings, and termination or censoring semantics. Preserve raw values;
   do not silently fill gaps, invent labels, or manufacture continuous time.
3. Implement selectors and inventory with consistent Polars schemas. Keep
   source tables separate when a join adds target information. For example,
   REFIT annotations must remain distinct from power measurements unless a
   caller explicitly requests a documented alignment.
4. Register the loader in `pdmdata/datasets/load.py`, add `metadata.json`, and
   generate dataset/catalog documentation. Mark native targets separately from
   derived targets. Task-specific labels, windows, entity splits, RUL endpoints,
   and train/test construction belong in `pdmdata/tasks/`, not the loader.
5. Validate parsing and selectors with representative fixtures, then compare
   at least one real provider file with the loader output. For multi-file data,
   check cross-file schema and entity coverage before claiming whole-dataset
   validation. Keep large downloads opt-in and out of Git.

## Standard dataset adapter layout

Each supported dataset lives under `pdmdata/datasets/<id>/` and follows a
consistent module, asset, and public API convention:

```text
pdmdata/datasets/<id>/
  __init__.py           # Public adapter API (load, inventory, verify, ...)
  loader.py             # Source-faithful Polars DataFrame / LazyFrame loader
  metadata.json         # Machine-readable catalog, download, and schema metadata
  README.md             # Dataset guide with generated metadata markers
  viz.py                # Plotly specs and dataset-specific plot helpers
  validation.py         # Optional offline source verification exposing verify()
  assets/               # Optional preview PNGs and validation/sample JSON files
```

### Package data and asset rules

- Raw data files (`.csv`, `.mat`, `.h5`, `.zip`, `.npz`, etc.) never live under
  `pdmdata/`; they belong under the configured `data_root/<id>/`
  (default: `data/raw/<id>/`).
- Any auxiliary JSON records (such as recording manifests or full-file
  validation reports) and compact README preview images belong in
  `pdmdata/datasets/<id>/assets/` so `pyproject.toml` includes them
  automatically via `datasets/*/assets/*.json` and `datasets/*/assets/*.png`.

## Step-by-step implementation checklist

1. **Metadata and loader**:
   - Create `pdmdata/datasets/<id>/metadata.json` with `id`, `name`,
     `short_name`, `summary`, `source`, `license`, `availability`,
     `characteristics`, `task_support`, `attributes`, and optional `download`
     and `citation`.
   - Implement `load(...)` (and optional `inventory(...)`) in
     `pdmdata/datasets/<id>/loader.py` and re-export them from
     `pdmdata/datasets/<id>/__init__.py`.
2. **Offline verification (when applicable)**:
   - Implement `verify(...)` in `pdmdata/datasets/<id>/validation.py`,
     re-export `verify` from `pdmdata/datasets/<id>/__init__.py`, and register
     the module in `_VERIFIER_MODULES` in `pdmdata/datasets/load.py`.
3. **Global dispatch and visualization**:
   - Register `pdmdata.datasets.<id>.loader` in `_LOADER_MODULES` in
     `pdmdata/datasets/load.py`.
   - Define `PLOTS` in `pdmdata/datasets/<id>/viz.py` and register it in
     `pdmdata/visualization/registry.py`.
4. **Documentation and notebooks**:
   - Create `pdmdata/datasets/<id>/README.md` and run
     `uv run --locked python scripts/generate_docs.py` to synchronize the
     generated metadata section and `pdmdata/README.md`.
   - Add `scripts/notebooks/datasets/generate_<id>_notebook.py`, generate
     `notebooks/datasets/<id>.ipynb` via
     `uv run --locked python scripts/notebooks/generate_all.py`, and add the
     entry to the table in `notebooks/datasets/README.md`.
5. **Tests and lint**:
   - Add unit tests in `tests/datasets/test_<id>.py` covering selectors,
     schema preservation, invalid-option handling, and verification.
   - Run the full test suite and linter:
     ```bash
     uv run --locked python -m unittest discover -s tests -p 'test_*.py' -q
     uv run --locked ruff check pdmdata tests scripts
     ```
