# Dataset catalog

Browse the datasets and task comparison below, then open a dataset README for
feature dimensions, source formats, selectors, and terms of use.

[Getting started](../README.md#getting-started) · [Usage guide](../docs/usage.md)

## Available datasets

The following table summarizes the available datasets. RUL means remaining
useful life.

<!-- BEGIN GENERATED DATASET TABLE -->

| Dataset | Data type | Tasks | License | Access |
|---|---|---|---|---|
| [Backblaze](backblaze/README.md) | Fleet-scale daily reliability panel | Drive failure prediction, Survival analysis, Time-to-event prediction | Backblaze Drive Stats terms | Direct quarterly downloads |
| [CARE](care/README.md) | Large labeled wind-turbine SCADA time series | Early fault detection, Anomaly detection, Failure prediction | CC BY-SA 4.0 | Direct download from Zenodo |
| [C-MAPSS](cmapss/README.md) | Multivariate run-to-failure time series | RUL estimation, Prognostics, Anomaly detection | Unspecified | Direct download from NASA |
| [GDD](gdd/README.md) | Multivariate time series | Anomaly detection, State classification | Requires verification | Kaggle |
| [GFD](gfd/README.md) | Vibration time series | Fault classification, Signal analysis | CC BY 4.0 | Direct download |
| [HydSys](hydsys/README.md) | Multirate multivariate time series | Fault classification, Condition estimation, Regression | CC BY 4.0 | Direct ZIP download from UCI |
| [IMS](ims/README.md) | Run-to-failure vibration time series | Fault diagnosis, Degradation estimation, Prognostics | U.S. Government Works | Direct NASA-linked S3 download; nested ZIP, 7z, and RAR extraction with libarchive |
| [MAPM](mapm/README.md) | Multi-table time series | Failure prediction, RUL estimation, Maintenance analysis | Unknown | Direct download from Microsoft GitHub |
| [MetroPT2](metropt2/README.md) | Large multivariate equipment time series | Online anomaly detection, Failure prediction, RUL research | CC BY 4.0 | Direct download from Zenodo |
| [OYICD](oyicd/README.md) | Multivariate degradation time series | Degradation estimation, Anomaly detection, Operating-mode classification, RUL research | CC BY-SA 3.0 | Kaggle |
| [PPD](ppd/README.md) | Run-to-failure time series | Degradation estimation, Anomaly detection, RUL research | CC BY-SA 3.0 | Kaggle |

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
| [Backblaze](backblaze/README.md) | △ | ✓ | — | △ | △ | ✓ | ✓ | — | — |
| [CARE](care/README.md) | ✓ | ✓ | — | △ | — | ✓ | △ | — | — |
| [C-MAPSS](cmapss/README.md) | △ | — | — | △ | ✓ | ✓ | ✓ | — | — |
| [GDD](gdd/README.md) | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| [GFD](gfd/README.md) | △ | ✓ | — | — | — | — | — | — | — |
| [HydSys](hydsys/README.md) | — | ✓ | — | ✓ | — | — | — | — | — |
| [IMS](ims/README.md) | △ | △ | — | △ | △ | △ | △ | — | — |
| [MAPM](mapm/README.md) | △ | ✓ | — | — | △ | ✓ | ✓ | ✓ | △ |
| [MetroPT2](metropt2/README.md) | △ | △ | — | △ | △ | △ | — | — | — |
| [OYICD](oyicd/README.md) | △ | — | ✓ | △ | △ | △ | △ | — | — |
| [PPD](ppd/README.md) | △ | — | — | △ | △ | △ | △ | — | — |

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

## Reference datasets

These resources are not currently supported by this project:

- [ALPI — Alarm Logs in Packaging Industry](https://data.mendeley.com/datasets/4nhx2x67cd/1): retained for reference; automated retrieval and the source-file layout have not been verified. CLI access returned HTTP 403 in our environment on 2026-09-06.

- [Oxford Battery Degradation Dataset 1](https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac)
- [Battery Degradation Dataset](https://data.mendeley.com/datasets/kw34hhw7xg/2)
- [Vega shrink-wrapper component degradation](https://www.kaggle.com/datasets/inIT-OWL/vega-shrinkwrapper-runtofailure-data)
- [CWRU Bearing Dataset mirror](https://www.kaggle.com/datasets/brjapon/cwru-bearing-datasets)
