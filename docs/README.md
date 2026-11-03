# Documentation index

Guides and architecture references for using and extending PdMData.

[Project overview](../README.md) · [Dataset catalog](../pdmdata/README.md) · [Task overview](../pdmdata/tasks/README.md)

## Guides in `docs/`

| Document | Contents |
|---|---|
| [Usage guide](usage.md) | Environment setup, `pdmdata.toml` configuration, Kaggle authentication, dataset downloads, Polars loading, Plotly visualization, and RUL/TTE task workflows |
| [Contributing dataset adapters](contributing-datasets.md) | Adapter acceptance criteria, standard `pdmdata/datasets/<id>/` layout, asset conventions, and step-by-step checklist for adding a dataset |
| [Meta-learning data guide](meta-learning-data.md) | Cross-condition and cross-machine adaptation boundaries, evaluation contract, and dataset selection |
| [Dataset candidates](dataset-candidates.md) | Candidate dataset reviews, adoption priorities, and candidate-specific implementation checks |

## Package and workflow indexes

| Area | Index | Description |
|---|---|---|
| Dataset catalog | [`pdmdata/README.md`](../pdmdata/README.md) | Supported datasets, dataset × task suitability matrix, and provider links |
| Task packages | [`pdmdata/tasks/README.md`](../pdmdata/tasks/README.md) | Canonical task taxonomy, [RUL guide](../pdmdata/tasks/rul/README.md), and [TTE guide](../pdmdata/tasks/tte/README.md) |
| Notebooks | [`notebooks/README.md`](../notebooks/README.md) | [Dataset notebooks](../notebooks/datasets/README.md) and [task notebooks](../notebooks/tasks/README.md) |
| CLI & maintainer scripts | [`scripts/README.md`](../scripts/README.md) | `pdmdata` CLI, bootstrap, dataset download scripts, documentation generation, notebook generation, and maintenance scripts |
