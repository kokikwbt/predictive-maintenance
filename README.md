# Predictive Maintenance

This repository is intended to enable quick access to datasets for predictive maintenance (PM) tasks (under development).
The following table summarizes the available features,
where the mark \* on dataset names shows
the richness of attributes you may check them up with higher priority.
Note that RUL means remaining useful life.

<!-- BEGIN GENERATED DATASET TABLE -->

| Dataset | Available | Data type | Tasks | License | Access |
|---|:---:|---|---|---|---|
| [ALPI](datasets/alpi/readme.md) | ✓ | Event sequence | Alarm forecasting, Anomaly detection, Sequence forecasting | CC BY 4.0 | Direct download |
| [CBM](datasets/cbm/README.md) | ✓ | Multivariate tabular | Condition estimation, Regression | CC BY 4.0 | Direct download / ucimlrepo |
| [C-MAPSS](datasets/cmapss/README.md) | ✓ | Multivariate run-to-failure time series | RUL estimation, Prognostics, Anomaly detection | Unspecified | Direct download from NASA |
| [GDD](datasets/gdd/README.md) | ✓ | Multivariate time series | Anomaly detection, State classification | Requires verification | Kaggle |
| [GFD](datasets/gfd/README.md) | ✓ | Vibration time series | Fault classification, Signal analysis | CC BY 4.0 | Direct download |
| [HydSys](datasets/hydsys/README.md) | ✓ | Multirate multivariate time series | Fault classification, Condition estimation, Regression | CC BY 4.0 | Direct download / ucimlrepo |
| [IMS](datasets/ims/README.md) | ✓ | Run-to-failure vibration time series | Fault diagnosis, Degradation estimation, Prognostics | U.S. Government Works | Direct download from NASA |
| [MAPM](datasets/mapm/README.md) | ✓ | Multi-table time series | Failure prediction, RUL estimation, Maintenance analysis | Unknown | Kaggle CLI |
| [OYICD](datasets/oyicd/README.md) | ✓ | Multivariate degradation time series | Degradation estimation, Anomaly detection, Operating-mode classification, RUL research | CC BY-SA 3.0 | Kaggle |
| [PPD](datasets/ppd/README.md) | ✓ | Run-to-failure time series | Degradation estimation, Anomaly detection, RUL research | CC BY-SA 3.0 | Kaggle |
| [UFD](datasets/ufd/README.md) | ✓ | Multivariate tabular | Fault classification, Condition diagnosis | CC BY 4.0 | Direct download / ucimlrepo |

<!-- END GENERATED DATASET TABLE -->

<!-- | NASA-B    |  |  |  |  | Other |
| CWRU-B    |  |  |  |  | CC-BY-SA | -->

## Installation

The recommended Conda environment name is `pmdata`, short for Predictive
Maintenance Data.

### Automated setup

From the repository root, run the bootstrap script:

```bash
./scripts/bootstrap.sh
```

This single command creates or reuses the `pmdata` Conda environment, installs
the Python requirements, registers the `Python (pmdata)` Jupyter kernel, and
downloads and extracts every dataset supported by direct URLs or the official
Kaggle CLI.
Existing archives are reused, so the command can be run again after an
interrupted setup.

Conda must already be installed and available in `PATH`. Source archives,
extracted files, and SHA-256 manifests are stored under `data/raw/<dataset-id>`.

### Manual setup

```bash
conda create --name pmdata python=3.11 pip
conda activate pmdata
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name pmdata --display-name "Python (pmdata)"
```

Start JupyterLab from the activated environment with `jupyter lab`, then select
`Python (pmdata)` as the notebook kernel.

Dataset loaders and preprocessing examples use Polars.

## Downloading datasets

The project supports direct URLs and Kaggle datasets:

- CBM
- C-MAPSS
- GDD (Kaggle)
- GFD
- HydSys
- MAPM (Kaggle)
- PPD (Kaggle)
- UFD

The repository does not implement authentication or handle credentials. Public
Kaggle datasets are attempted through the official CLI. If Kaggle requires
authentication, the user must complete it independently and retry:

```bash
kaggle auth login
```

Credentials remain under Kaggle CLI control and must not be added to this
repository.

Download all supported datasets without repeating the environment setup:

```bash
conda activate pmdata
python scripts/download_all.py
```

Download and extract a dataset into `data/raw/<dataset-id>`:

```bash
python scripts/download.py hydsys
```

Print the configured external command without downloading:

```bash
python scripts/download.py hydsys --print-command
python scripts/download.py gdd --print-command
```

The downloader preserves the source archive, extracts it into a separate
directory, validates expected files, and records the archive SHA-256 in
`manifest.json`. The `data/` directory is excluded from Git.

ALPI is not currently supported by the downloader.

## Usage

Import the package from the repository root:

```python
import datasets

# Display the dataset catalog
datasets.summary()

# Download a supported dataset into data/raw
datasets.download("ufd")

# Load downloaded source data as a Polars DataFrame
frame = datasets.load("ufd", meter="A")
```

Show only datasets supported by an automated download method:

```python
datasets.summary(downloadable=True)
```

Dataset files are not stored under `datasets/`. That directory contains
metadata and analysis-library code; all downloaded source data belongs under
`data/raw/<dataset-id>`. Dataset loaders and notebooks are being rebuilt around
this layout.

## Analysis conventions

### Run-to-failure data

Run-to-failure data require:

- time column
- event/censoring column (categorical)
- numerical/categorical feature columns (optional)

## Notebooks

Jupyter notebooks are organized by purpose:

- [`notebooks/datasets/`](notebooks/datasets/) is reserved for rebuilt loading,
  inspection, and visualization notebooks.
- [`notebooks/tasks/`](notebooks/tasks/) contains predictive-maintenance
  analysis examples that can apply across datasets.

See the [notebook guide](notebooks/README.md) for the organization and
contribution guidelines.


## Recommended projects

- [IBM FailureSensorIQ](https://github.com/IBM/FailureSensorIQ): a dataset and
  benchmark for evaluating language-model reasoning about relationships
  between industrial sensor behavior and failure modes.
- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench): a framework and
  benchmark for building, orchestrating, and evaluating AI agents for
  industrial asset operations and maintenance.

## References

### Predictive-maintenance background

- [Predictive maintenance overview](https://en.wikipedia.org/wiki/Predictive_maintenance)
- [Azure architecture guide for predictive-maintenance solutions](https://learn.microsoft.com/azure/architecture/data-science-process/predictive-maintenance-playbook)
- [Types of proactive maintenance](https://solutions.borderstates.com/types-of-proactive-maintenance/)
- [Kaggle community guide to common dataset licenses](https://www.kaggle.com/general/116302)

### Dataset Sources

- [ALPI — Alarm Logs in Packaging Industry](https://ieee-dataport.org/open-access/alarm-logs-packaging-industry-alpi)
- [CBM — Condition Based Maintenance of Naval Propulsion Plants](https://archive.ics.uci.edu/dataset/316/condition+based+maintenance+of+naval+propulsion+plants)
- [C-MAPSS — NASA Prognostics Center of Excellence data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [GDD — Genesis Demonstrator Data for Machine Learning](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning)
- [GFD — Gearbox Fault Diagnosis](https://openei.org/datasets/dataset/gearbox-fault-diagnosis-data)
- [HydSys — Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- [IMS — IMS Bearings](https://catalog.data.gov/dataset/ims-bearings)
- [MAPM — Microsoft Azure Predictive Maintenance mirror](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
- [OYICD — One Year Industrial Component Degradation](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation)
- [PPD — Production Plant Data for Condition Monitoring](https://www.kaggle.com/datasets/inIT-OWL/production-plant-data-for-condition-monitoring)
- [UFD — Ultrasonic Flowmeter Diagnostics](https://archive.ics.uci.edu/dataset/433/ultrasonic+flowmeter+diagnostics)

### Candidate datasets

These resources are not currently part of the catalog:

- [Oxford Battery Degradation Dataset 1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)
- [Battery Degradation Dataset](https://data.mendeley.com/datasets/kw34hhw7xg/2)
- [Vega shrink-wrapper component degradation](https://www.kaggle.com/datasets/inIT-OWL/vega-shrinkwrapper-runtofailure-data)
- [CWRU Bearing Dataset mirror](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)


## License

Repository code and documentation are available under the MIT License. Each
dataset remains subject to its own license and terms of use; consult its
metadata and source before use.
