# Dataset catalog

Browse the datasets and task comparison below, then open a dataset README for
feature dimensions, source formats, selectors, and terms of use. Implemented
task packages are documented under [tasks/](tasks/README.md).

[Getting started](../README.md#getting-started) · [Usage guide](../docs/usage.md)

## Available datasets

The following table summarizes the available datasets. RUL means remaining
useful life.

<!-- BEGIN GENERATED DATASET TABLE -->

| Dataset | Data type | Tasks | License | Access |
|---|---|---|---|---|
| [ALPI](datasets/alpi/README.md) | Industrial alarm event sequences | Next-alarm prediction, Future-window alarm prediction, Event-sequence forecasting, Anomaly detection research | CC BY 4.0 | Direct Mendeley file download; CLI transfer and provider SHA-256 verified |
| [Backblaze](datasets/backblaze/README.md) | Fleet-scale daily reliability panel | Drive failure prediction, Survival analysis, Time-to-event prediction | Backblaze Drive Stats terms | Direct quarterly downloads |
| [CARE](datasets/care/README.md) | Large labeled wind-turbine SCADA time series | Early fault detection, Anomaly detection, Failure prediction | CC BY-SA 4.0 | Direct download from Zenodo |
| [C-MAPSS](datasets/cmapss/README.md) | Multivariate run-to-failure time series | RUL estimation, Prognostics, Anomaly detection | Unspecified | Direct download from NASA |
| [GDD](datasets/gdd/README.md) | Multivariate time series | Anomaly detection, State classification | Requires verification | Kaggle |
| [GFD](datasets/gfd/README.md) | Vibration time series | Fault classification, Signal analysis | CC BY 4.0 | Direct download |
| [HydSys](datasets/hydsys/README.md) | Multirate multivariate time series | Fault classification, Condition estimation, Regression | CC BY 4.0 | Direct ZIP download from UCI |
| [IMS](datasets/ims/README.md) | Run-to-failure vibration time series | Fault diagnosis, Degradation estimation, Prognostics | U.S. Government Works | Direct NASA-linked S3 download; nested ZIP, 7z, and RAR extraction with libarchive |
| [MAPM](datasets/mapm/README.md) | Multi-table time series | Failure prediction, RUL estimation, Maintenance analysis | Unknown | Direct download from Microsoft GitHub |
| [MetroPT2](datasets/metropt2/README.md) | Large multivariate equipment time series | Online anomaly detection, Failure prediction, RUL research | CC BY 4.0 | Direct download from Zenodo |
| [N-CMAPSS](datasets/ncmapss/README.md) | Multivariate run-to-failure time series under real flight conditions | RUL estimation, Prognostics, Fault diagnostics | U.S. Government Works / provider terms | Per-subset HDF5 downloads via Kaggle mirror (official NASA bundle is a 15 GB nested ZIP) |
| [OYICD](datasets/oyicd/README.md) | Multivariate degradation time series | Degradation estimation, Anomaly detection, Operating-mode classification, RUL research | CC BY-SA 3.0 | Kaggle |
| [PPD](datasets/ppd/README.md) | Run-to-failure time series | Degradation estimation, Anomaly detection, RUL research | CC BY-SA 3.0 | Kaggle |
| [Scania Component X](datasets/scania_x/README.md) | Fleet-scale irregular multivariate time series with repair records | Imminent-failure classification, Time-to-event prediction, Survival analysis | CC BY 4.0 | Direct per-file CSV downloads from Researchdata.se (~1.65 GB; no account required) |
| [XJTU-SY](datasets/xjtu_sy/README.md) | Run-to-failure vibration time series | RUL estimation, Prognostics, Fault diagnostics | Public research use (citation requested) | Direct ZIP download via Hugging Face mirror of the official bundle (~5.4 GB) |

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
- **[Remaining useful life prediction](tasks/rul/README.md):** Predict remaining cycles or time before a defined failure endpoint.
- **[Time-to-event prediction](tasks/tte/README.md):** Predict when a failure, alarm, or maintenance-relevant event will occur.
- **Survival analysis:** Model event-time distributions while explicitly accounting for censoring.
- **Event or sequence forecasting:** Predict the type or sequence of future alarms, errors, or machine events.
- **Maintenance-policy evaluation:** Compare or learn maintenance decisions using intervention and outcome histories.

The matrix distinguishes immediately usable targets from tasks that require
label, endpoint, or health-index construction. Marks describe **dataset
suitability**, not whether an experiment package under `tasks/` already
implements `prepare_*` for that dataset. RUL prepares C-MAPSS,
N-CMAPSS, and XJTU-SY. TTE prepares those fleets and Backblaze,
CARE, MAPM, and Scania Component X. See
[tasks/README.md](tasks/README.md).

- ✓: directly supported by labels, targets, or event/censoring records
- △: supported after deriving labels or targets from chronology or domain assumptions
- —: not a natural use of the dataset

| Dataset | Anomaly | Fault class. | State class. | Condition | RUL | TTE | Survival | Event forecast | Maintenance |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [ALPI](datasets/alpi/README.md) | △ | — | — | — | — | △ | — | ✓ | — |
| [Backblaze](datasets/backblaze/README.md) | △ | — | — | △ | △ | ✓ | ✓ | — | — |
| [CARE](datasets/care/README.md) | ✓ | ✓ | — | △ | — | ✓ | △ | — | — |
| [C-MAPSS](datasets/cmapss/README.md) | △ | — | — | △ | ✓ | ✓ | ✓ | — | — |
| [GDD](datasets/gdd/README.md) | ✓ | — | ✓ | — | — | — | — | — | — |
| [GFD](datasets/gfd/README.md) | △ | ✓ | — | — | — | — | — | — | — |
| [HydSys](datasets/hydsys/README.md) | — | ✓ | — | ✓ | — | — | — | — | — |
| [IMS](datasets/ims/README.md) | △ | △ | — | △ | △ | △ | △ | — | — |
| [MAPM](datasets/mapm/README.md) | △ | ✓ | — | — | △ | ✓ | ✓ | ✓ | △ |
| [MetroPT2](datasets/metropt2/README.md) | △ | △ | — | △ | △ | △ | — | — | — |
| [N-CMAPSS](datasets/ncmapss/README.md) | △ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — |
| [OYICD](datasets/oyicd/README.md) | △ | — | ✓ | △ | △ | △ | △ | — | — |
| [PPD](datasets/ppd/README.md) | △ | — | — | △ | △ | △ | △ | — | — |
| [Scania Component X](datasets/scania_x/README.md) | △ | — | — | △ | △ | ✓ | ✓ | — | — |
| [XJTU-SY](datasets/xjtu_sy/README.md) | △ | △ | — | △ | ✓ | ✓ | △ | — | — |

<!-- END GENERATED TASK TABLE -->

## Dataset sources

<!-- BEGIN GENERATED DATASET SOURCES -->

- [ALPI — Alarm Logs in Packaging Industry](https://data.mendeley.com/datasets/4nhx2x67cd/1)
- [Backblaze — Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
- [CARE — Wind Turbine SCADA Data for Early Fault Detection](https://zenodo.org/records/15846963)
- [C-MAPSS — Turbofan Engine Degradation Simulation Data Set](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [GDD — Genesis Demonstrator Data for Machine Learning](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning)
- [GFD — Gearbox Fault Diagnosis Data](https://data.openei.org/submissions/623)
- [HydSys — Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- [IMS — IMS Bearings](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [MAPM — Microsoft Azure Predictive Maintenance](https://github.com/microsoft/sqlworkshops/tree/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data)
- [MetroPT2 — MetroPT2: A Benchmark Dataset for Predictive Maintenance](https://zenodo.org/records/7766691)
- [N-CMAPSS — Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions](https://data.phmsociety.org/nasa/)
- [OYICD — One Year Industrial Component Degradation](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation)
- [PPD — Production Plant Data for Condition Monitoring](https://www.kaggle.com/datasets/inIT-OWL/production-plant-data-for-condition-monitoring)
- [Scania Component X — SCANIA Component X Dataset](https://researchdata.se/en/catalogue/dataset/2024-34)
- [XJTU-SY — XJTU-SY Bearing Datasets](https://biaowang.tech/xjtu-sy-bearing-datasets/)

<!-- END GENERATED DATASET SOURCES -->

## Reference links

Related datasets and benchmarks that are **not** currently supported as
PdMData adapters. Listed for discovery only; licenses and access terms are
those of each source. See the [candidate review](../docs/dataset-candidates.md)
for proposed tasks, priorities, and the checks required before an adapter is
registered.

### Industrial sensor / degradation data

- [FEMTO / PRONOSTIA (IEEE PHM 2012)](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset)
  — Accelerated bearing run-to-failure challenge corpus (FEMTO-ST /
  PRONOSTIA; original challenge page offline; community mirror).
- [CWRU Bearing Dataset mirror](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)
  — Classic bearing vibration fault-classification corpus
  (Case Western Reserve University; Kaggle mirror).
- [Oxford Battery Degradation Dataset 1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)
  — Lab battery aging trajectories under controlled cycling
  (University of Oxford).
- [Battery Degradation Dataset](https://data.mendeley.com/datasets/kw34hhw7xg/2)
  — Battery degradation under fixed-current and arbitrary-use profiles
  (Mendeley Data).
- [Vega shrink-wrapper component degradation](https://www.kaggle.com/datasets/inIT-OWL/vega-shrinkwrapper-runtofailure-data)
  — Run-to-failure component data from an industrial shrink-wrapper
  (inIT / OWL).

- [REFIT electrical measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/)
  with [annotated load anomalies](https://pureportal.strath.ac.uk/en/datasets/annotated-load-anomalies-from-the-refit-dataset/)
  — Appliance condition-monitoring candidate with separate power measurements
  and anomaly annotations. No adapter yet; load anomalies are not confirmed
  failure or repair endpoints.

### LLM / agent benchmarks (PdM-related)

These target language-model or multi-agent evaluation rather than Polars
tabular loaders, so they stay outside the supported catalog.

- [IBM FailureSensorIQ](https://github.com/IBM/FailureSensorIQ)
  — Multi-choice QA benchmark on sensor–failure-mode relationships for
  industrial assets (Hugging Face dataset + leaderboard).
- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench)
  — Agent / multi-agent benchmark for industrial asset operations and
  maintenance (scenarios, IoT simulation, PHM-related tasks).
