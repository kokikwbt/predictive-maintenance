# PdMData: Datasets for Predictive Maintenance

A Python toolkit to download, load, and visualize predictive-maintenance
datasets for research. Use a shared API with dataset-specific loaders; download
only what you need.

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

Open a dataset README from the catalog for selectors, schema details, and
terms of use. Open a task README for experiment bundles and scoring
protocols.

## License

Repository code and documentation are available under the MIT License. Each
dataset remains subject to its own license and terms of use; consult its
README and source before use.
