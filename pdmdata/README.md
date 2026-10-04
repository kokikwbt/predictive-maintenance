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
| [Scania Component X](datasets/scania_x/README.md) | Fleet-scale irregular multivariate time series with repair records | Imminent-failure classification, Time-to-event prediction, Survival analysis, Cost-sensitive maintenance decisions | CC BY 4.0 | Direct per-file CSV downloads from Researchdata.se (~1.65 GB; no account required) |
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
label, endpoint, or health-index construction.

- ✓: directly supported by labels, targets, or event/censoring records
- △: supported after deriving labels or targets from chronology or domain assumptions
- —: not a natural use of the dataset

| Dataset | Anomaly | Fault class. | State class. | Condition | RUL | TTE | Survival | Event forecast | Maintenance |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [Backblaze](datasets/backblaze/README.md) | △ | ✓ | — | △ | △ | ✓ | ✓ | — | — |
| [CARE](datasets/care/README.md) | ✓ | ✓ | — | △ | — | ✓ | △ | — | — |
| [C-MAPSS](datasets/cmapss/README.md) | △ | — | — | △ | ✓ | ✓ | ✓ | — | — |
| [GDD](datasets/gdd/README.md) | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| [GFD](datasets/gfd/README.md) | △ | ✓ | — | — | — | — | — | — | — |
| [HydSys](datasets/hydsys/README.md) | — | ✓ | — | ✓ | — | — | — | — | — |
| [IMS](datasets/ims/README.md) | △ | △ | — | △ | △ | △ | △ | — | — |
| [MAPM](datasets/mapm/README.md) | △ | ✓ | — | — | △ | ✓ | ✓ | ✓ | △ |
| [MetroPT2](datasets/metropt2/README.md) | △ | △ | — | △ | △ | △ | — | — | — |
| [N-CMAPSS](datasets/ncmapss/README.md) | △ | △ | — | △ | ✓ | ✓ | ✓ | — | — |
| [OYICD](datasets/oyicd/README.md) | △ | — | ✓ | △ | △ | △ | △ | — | — |
| [PPD](datasets/ppd/README.md) | △ | — | — | △ | △ | △ | △ | — | — |
| [Scania Component X](datasets/scania_x/README.md) | △ | ✓ | — | △ | △ | ✓ | ✓ | — | △ |
| [XJTU-SY](datasets/xjtu_sy/README.md) | △ | △ | — | △ | ✓ | ✓ | ✓ | — | — |

<!-- END GENERATED TASK TABLE -->

## Dataset sources

- [C-MAPSS — NASA Prognostics Center of Excellence data repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [GDD — Genesis Demonstrator Data for Machine Learning](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning)
- [GFD — Gearbox Fault Diagnosis](https://openei.org/datasets/dataset/gearbox-fault-diagnosis-data)
- [HydSys — Condition Monitoring of Hydraulic Systems](https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems)
- [IMS — IMS Bearings](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [MAPM — Microsoft Azure Predictive Maintenance](https://github.com/microsoft/sqlworkshops/tree/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data)
- [MetroPT2 — Metro do Porto compressor data](https://zenodo.org/records/7766691)
- [OYICD — One Year Industrial Component Degradation](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation)
- [PPD — Production Plant Data for Condition Monitoring](https://www.kaggle.com/datasets/inIT-OWL/production-plant-data-for-condition-monitoring)
- [CARE to Compare — wind-turbine fault-detection benchmark](https://zenodo.org/records/15846963)
- [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
- [SCANIA Component X — Researchdata.se (SND)](https://researchdata.se/en/catalogue/dataset/2024-34)

## Reference links

Related datasets and benchmarks that are **not** currently supported as
PdMData adapters. Listed for discovery only; licenses and access terms are
those of each source.

### Industrial sensor / degradation data

- [ALPI — Alarm Logs in Packaging Industry](https://data.mendeley.com/datasets/4nhx2x67cd/1)
  — Packaging-line alarm logs for event / sequence analysis. Automated
  retrieval and layout have not been verified (HTTP 403 on 2026-09-06).
- [Oxford Battery Degradation Dataset 1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)
  — Lab battery aging trajectories under controlled cycling
  (University of Oxford).
- [Battery Degradation Dataset](https://data.mendeley.com/datasets/kw34hhw7xg/2)
  — Battery degradation under fixed-current and arbitrary-use profiles
  (Mendeley Data).
- [Vega shrink-wrapper component degradation](https://www.kaggle.com/datasets/inIT-OWL/vega-shrinkwrapper-runtofailure-data)
  — Run-to-failure component data from an industrial shrink-wrapper
  (inIT / OWL).
- [CWRU Bearing Dataset mirror](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)
  — Classic bearing vibration fault-classification corpus
  (Case Western Reserve University; Kaggle mirror).

### LLM / agent benchmarks (PdM-related)

These target language-model or multi-agent evaluation rather than Polars
tabular loaders, so they stay outside the supported catalog.

- [IBM FailureSensorIQ](https://github.com/IBM/FailureSensorIQ)
  — Multi-choice QA benchmark on sensor–failure-mode relationships for
  industrial assets (Hugging Face dataset + leaderboard).
- [IBM AssetOpsBench](https://github.com/IBM/AssetOpsBench)
  — Agent / multi-agent benchmark for industrial asset operations and
  maintenance (scenarios, IoT simulation, PHM-related tasks).
