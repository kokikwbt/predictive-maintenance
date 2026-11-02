# Documentation index

Guides and architecture references for using and extending PdMData.

[Project overview](../README.md) · [Dataset catalog](../pdmdata/README.md) · [Task overview](../pdmdata/tasks/README.md)

## Guides in `docs/`

| Document | Contents |
|---|---|
| [Usage guide](usage.md) | Environment setup, `pdmdata.toml` configuration, Kaggle authentication, dataset downloads, Polars loading, Plotly visualization, and RUL/TTE task workflows |
| [Meta-learning data guide](meta-learning-data.md) | Cross-condition and cross-machine adaptation boundaries, evaluation contract, and dataset selection |
| [Dataset candidates](dataset-candidates.md) | Candidate dataset reviews, adoption priorities, and [adapter acceptance criteria](dataset-candidates.md#adapter-acceptance-criteria) |

## Package and workflow indexes

| Area | Index | Description |
|---|---|---|
| Dataset catalog | [`pdmdata/README.md`](../pdmdata/README.md) | Supported datasets, dataset × task suitability matrix, and provider links |
| Task packages | [`pdmdata/tasks/README.md`](../pdmdata/tasks/README.md) | Canonical task taxonomy, [RUL guide](../pdmdata/tasks/rul/README.md), and [TTE guide](../pdmdata/tasks/tte/README.md) |
| Notebooks | [`notebooks/README.md`](../notebooks/README.md) | [Dataset notebooks](../notebooks/datasets/README.md) and [task notebooks](../notebooks/tasks/README.md) |
| CLI & maintainer scripts | [`scripts/README.md`](../scripts/README.md) | Bootstrap, dataset download CLIs, documentation generation, notebook generation, and maintenance scripts |

## Standard dataset adapter layout

Each supported dataset lives under `pdmdata/datasets/<id>/` and follows a
consistent module and asset convention:

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

### Adding or updating a dataset adapter

1. Review the [adapter acceptance criteria](dataset-candidates.md#adapter-acceptance-criteria)
   to verify license terms, checksums, and source-faithful schema semantics.
2. Create `pdmdata/datasets/<id>/metadata.json` and `loader.py`, keeping raw
   data under `data/raw/<id>/` and placing any bundled preview images or JSON
   records inside `pdmdata/datasets/<id>/assets/`.
3. Expose `load` (and optional `inventory` / `verify`) in
   `pdmdata/datasets/<id>/__init__.py`, and register the loader in
   `pdmdata/datasets/load.py`.
4. Register Plotly specifications in `pdmdata/datasets/<id>/viz.py` and
   `pdmdata/visualization/registry.py`.
5. Run `uv run --locked python scripts/generate_docs.py` to synchronize
   `pdmdata/datasets/<id>/README.md` and `pdmdata/README.md`.
6. Add unit tests under `tests/datasets/test_<id>.py` and run the test suite
   and linter (`uv run --locked python -m unittest discover -s tests -q` and
   `uv run --locked ruff check pdmdata tests scripts`).
