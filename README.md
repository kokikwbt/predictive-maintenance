# Predictive Maintenance

This repository is intended to enable quick access to datasets for predictive maintenance (PM) tasks (under development).
The following table summarizes the available datasets. RUL means remaining
useful life.

<!-- BEGIN GENERATED DATASET TABLE -->

| Dataset | Available | Data type | Tasks | License | Access |
|---|:---:|---|---|---|---|
| [ALPI](datasets/alpi/readme.md) | ✓ | Event sequence | Alarm forecasting, Anomaly detection, Sequence forecasting | CC BY 4.0 | Direct download |
| [Backblaze](datasets/backblaze/README.md) | ✓ | Fleet-scale daily reliability panel | Drive failure prediction, Survival analysis, Time-to-event prediction | Backblaze Drive Stats terms | Direct quarterly downloads |
| [CARE](datasets/care/README.md) | ✓ | Large labeled wind-turbine SCADA time series | Early fault detection, Anomaly detection, Failure prediction | CC BY-SA 4.0 | Direct download from Zenodo |
| [CBM](datasets/cbm/README.md) | ✓ | Multivariate tabular | Condition estimation, Regression | CC BY 4.0 | Direct download / ucimlrepo |
| [C-MAPSS](datasets/cmapss/README.md) | ✓ | Multivariate run-to-failure time series | RUL estimation, Prognostics, Anomaly detection | Unspecified | Direct download from NASA |
| [GDD](datasets/gdd/README.md) | ✓ | Multivariate time series | Anomaly detection, State classification | Requires verification | Kaggle |
| [GFD](datasets/gfd/README.md) | ✓ | Vibration time series | Fault classification, Signal analysis | CC BY 4.0 | Direct download |
| [HydSys](datasets/hydsys/README.md) | ✓ | Multirate multivariate time series | Fault classification, Condition estimation, Regression | CC BY 4.0 | Direct download / ucimlrepo |
| [IMS](datasets/ims/README.md) | ✓ | Run-to-failure vibration time series | Fault diagnosis, Degradation estimation, Prognostics | U.S. Government Works | Direct download from NASA |
| [MAPM](datasets/mapm/README.md) | ✓ | Multi-table time series | Failure prediction, RUL estimation, Maintenance analysis | Unknown | Kaggle CLI |
| [MetroPT2](datasets/metropt2/README.md) | ✓ | Large multivariate equipment time series | Online anomaly detection, Failure prediction, RUL research | CC BY 4.0 | Direct download from Zenodo |
| [OYICD](datasets/oyicd/README.md) | ✓ | Multivariate degradation time series | Degradation estimation, Anomaly detection, Operating-mode classification, RUL research | CC BY-SA 3.0 | Kaggle |
| [PPD](datasets/ppd/README.md) | ✓ | Run-to-failure time series | Degradation estimation, Anomaly detection, RUL research | CC BY-SA 3.0 | Kaggle |
| [UFD](datasets/ufd/README.md) | ✓ | Multivariate tabular | Fault classification, Condition diagnosis | CC BY 4.0 | Direct download / ucimlrepo |

<!-- END GENERATED DATASET TABLE -->

## Predictive-maintenance tasks

Time-to-event prediction is the broad task of predicting when an event will
occur. Survival analysis is a time-to-event approach designed to model
censored observations, so it is listed separately when a dataset can support
that experimental design.

<!-- BEGIN GENERATED TASK TABLE -->

- **Anomaly detection:** Detect observations or sequences that depart from normal operation.
- **Fault or health-state classification:** Assign a discrete fault type, component state, or healthy/faulty label.
- **Operating-state classification:** Identify operating modes or machine states that are not themselves faults.
- **Condition estimation:** Estimate a continuous health indicator, degradation level, or component condition.
- **Remaining useful life prediction:** Predict remaining cycles or time before a defined failure endpoint.
- **Time-to-event prediction:** Predict when a failure, alarm, or maintenance-relevant event will occur.
- **Survival analysis:** Model event-time distributions while explicitly accounting for censoring.
- **Event or sequence forecasting:** Predict the type or sequence of future alarms, errors, or machine events.
- **Maintenance-policy evaluation:** Compare or learn maintenance decisions using intervention and outcome histories.

The matrix distinguishes immediately usable targets from tasks that require
label, endpoint, or health-index construction.

- ✓: directly supported by labels, targets, or event/censoring records
- △: supported after deriving labels or targets from chronology or domain assumptions
- —: not a natural use of the dataset

| Dataset | Anomaly | Fault class. | State class. | Condition | RUL | TTE | Survival | Event forecast | Maintenance |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [ALPI](datasets/alpi/readme.md) | △ | — | — | — | — | ✓ | △ | ✓ | — |
| [Backblaze](datasets/backblaze/README.md) | △ | ✓ | — | △ | △ | ✓ | ✓ | — | — |
| [CARE](datasets/care/README.md) | ✓ | ✓ | — | △ | — | ✓ | △ | — | — |
| [CBM](datasets/cbm/README.md) | — | — | — | ✓ | — | — | — | — | — |
| [C-MAPSS](datasets/cmapss/README.md) | △ | — | — | △ | ✓ | ✓ | ✓ | — | — |
| [GDD](datasets/gdd/README.md) | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| [GFD](datasets/gfd/README.md) | △ | ✓ | — | — | — | — | — | — | — |
| [HydSys](datasets/hydsys/README.md) | — | ✓ | — | ✓ | — | — | — | — | — |
| [IMS](datasets/ims/README.md) | △ | △ | — | △ | △ | △ | △ | — | — |
| [MAPM](datasets/mapm/README.md) | △ | ✓ | — | — | △ | ✓ | ✓ | ✓ | △ |
| [MetroPT2](datasets/metropt2/README.md) | △ | △ | — | △ | △ | △ | — | — | — |
| [OYICD](datasets/oyicd/README.md) | △ | — | ✓ | △ | △ | △ | △ | — | — |
| [PPD](datasets/ppd/README.md) | △ | — | — | △ | △ | △ | △ | — | — |
| [UFD](datasets/ufd/README.md) | △ | ✓ | — | — | — | — | — | — | — |

<!-- END GENERATED TASK TABLE -->

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
downloads and extracts the bootstrap-sized datasets supported by direct URLs
or the official Kaggle CLI. Multi-gigabyte datasets are opt-in.
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
- IMS
- MAPM (Kaggle)
- OYICD (Kaggle)
- PPD (Kaggle)
- UFD

Large, opt-in downloads are also supported:

- MetroPT2 (1.2 GB CSV)
- CARE to Compare (5.5 GB ZIP)
- Backblaze Drive Stats (one selected quarter; more than 10 GB extracted)

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

Download a large dataset explicitly:

```bash
python scripts/download.py metropt2
python scripts/download.py care
python scripts/download.py backblaze --variant 2025-q1
```

Backblaze variants currently include `2025-q1`, `2025-q2`, `2025-q3`, and
`2025-q4`. These large datasets are intentionally excluded from
`download_all.py` and `bootstrap.sh`.

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

The large tabular datasets use lazy Polars scans by default:

```python
metro = datasets.load("metropt2")  # pl.LazyFrame
care_event = datasets.load("care", recording="path/to/event.csv")
drives = datasets.load("backblaze", variant="2025-q1")
```

Use `.select(...)`, `.filter(...)`, and then `.collect()` so that Polars reads
only the required data. CARE loads one recording at a time because the archive
contains many separate event and normal-operation datasets.

Create a registered interactive Plotly figure:

```python
frame = datasets.load("cmapss", subset="FD001", split="train")
figure = datasets.visualize(
    "cmapss",
    "sensor_trajectory",
    frame,
    entity=1,
)
figure.show()
```

See the [visualization showcase](notebooks/datasets/visualization-showcase.ipynb)
for time-series, distribution, and event-timeline examples.

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
- [MetroPT2 — Metro do Porto compressor data](https://zenodo.org/records/7766691)
- [OYICD — One Year Industrial Component Degradation](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation)
- [PPD — Production Plant Data for Condition Monitoring](https://www.kaggle.com/datasets/inIT-OWL/production-plant-data-for-condition-monitoring)
- [UFD — Ultrasonic Flowmeter Diagnostics](https://archive.ics.uci.edu/dataset/433/ultrasonic+flowmeter+diagnostics)
- [CARE to Compare — wind-turbine fault-detection benchmark](https://zenodo.org/records/15846963)
- [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)

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
