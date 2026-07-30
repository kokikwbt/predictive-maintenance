# Genesis Demonstrator Data for Machine Learning

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

Time-series data from a portable pick-and-place demonstrator, including continuous signals, discrete states, and anomaly labels.

| Item | Details |
|---|---|
| ID | `gdd` |
| Name | Genesis Demonstrator Data for Machine Learning |
| Provider | [inIT-OWL on Kaggle](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning) |
| DOI | — |
| Availability | available (checked: 2026-07-24) |
| Access | Kaggle |
| License | Requires verification |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Multivariate time series |
| Tasks | Anomaly detection, State classification |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Direct |
| Fault or health-state classification | Direct |
| Operating-state classification | Direct |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `Timestamp` | Datetime | timestamp | Observation time recorded as a Unix timestamp. |
| `MotorData.SetCurrent / ActCurrent` | Float64 | sensor | Motor-current setpoint and measured value. |
| `MotorData.SetPosition / ActPosition` | Float64 | sensor | Motor-position setpoint and measured value. |
| `MotorData.SetSpeed / ActSpeed` | Float64 | sensor | Motor-speed setpoint and measured value. |
| `MotorData.SetAcceleration / IsAcceleration` | Float64 | sensor | Acceleration setpoint and measured value. |
| `MotorData.SetForce / IsForce` | Float64 | sensor | Force setpoint and measured value. |
| `MotorData.*, PLC_PRG.*, NVL_*` | Boolean/Integer | machine-state | Discrete state signals for position, gripping, material, and storage mechanisms. |
| `Label` | Integer | target | Labels for normal operation and two anomaly states; present only in some files. |

## Usage notes

- Distribution depends on Kaggle.
- The column set and presence of the Label column vary by file.

## Download

```bash
python scripts/download.py gdd
```

Kaggle dataset: [`inIT-OWL/genesis-demonstrator-data-for-machine-learning`](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning)

## Suggested citation

Schuster, R., Moriz, N. and von Birgelen, A. (2018). Genesis demonstrator data for machine learning.

<!-- END GENERATED METADATA -->
