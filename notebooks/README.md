# Notebooks

The notebooks are organized by purpose:

- [`datasets/`](datasets/): the location reserved for rebuilt dataset-specific
  loading, inspection, and visualization notebooks.
- [`tasks/`](tasks/): analysis examples organized around predictive-maintenance
  tasks rather than a particular dataset.

Run notebooks with the repository root as the working directory, or keep the
default notebook working directory. Dataset notebooks must load downloaded
files from `../../data/raw/<dataset-id>`.

After installing the project requirements, register the Conda environment as a
Jupyter kernel:

```bash
conda activate pmdata
python -m ipykernel install --user --name pmdata --display-name "Python (pmdata)"
jupyter lab
```

Select `Python (pmdata)` as the kernel when opening a notebook.

Generated figures and reports should be written outside the notebook
directories. Notebook output cells may be cleared before committing when their
contents are large or environment-specific.
