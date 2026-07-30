# Wind Turbine SCADA Data for Early Fault Detection

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

A real-world early-fault-detection benchmark containing 89 years of SCADA time series from 36 wind turbines across three wind farms.

| Item | Details |
|---|---|
| ID | `care` |
| Name | Wind Turbine SCADA Data for Early Fault Detection |
| Provider | [Fraunhofer IEE on Zenodo](https://zenodo.org/records/15846963) |
| DOI | [10.5281/zenodo.15846963](https://doi.org/10.5281/zenodo.15846963) |
| Availability | available (checked: 2026-07-25) |
| Access | Direct download from Zenodo |
| License | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Large labeled wind-turbine SCADA time series |
| Tasks | Early fault detection, Anomaly detection, Failure prediction |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Direct |
| Fault or health-state classification | Direct |
| Condition estimation | Requires target derivation |
| Time-to-event prediction | Direct |
| Survival analysis | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `timestamp` | Datetime | timestamp | SCADA observation time within an anonymized turbine event series. |
| `sensor statistics` | Float64 | sensor | Per-timestamp average, minimum, maximum, and standard-deviation statistics; feature counts differ by wind farm. |
| `status_type_id` | Categorical | target | Turbine-status label used for training-data quality filtering and anomaly evaluation. |
| `event metadata` | Mixed | event | Fault descriptions, event timing, asset identifiers, and prediction intervals supplied with event series. |

## Usage notes

- Version 6 corrects labels, timestamps, units, and event descriptions from earlier versions.
- The collection has 95 datasets: 45 lead up to labeled turbine faults and 50 represent normal behavior.
- The provider recommends using average signals cautiously and documents implausible Min, Max, and Std values for some features.
- The 5.5 GB archive is opt-in and is not downloaded by scripts/bootstrap.sh.

## Download

```bash
python scripts/download.py care
```

Source archive: [CARE_To_Compare.zip](https://zenodo.org/records/15846963/files/CARE_To_Compare.zip?download=1)

## Suggested citation

Gück, C. and Roelofs, C. (2025). Wind Turbine SCADA Data For Early Fault Detection, Version 6. Zenodo. https://doi.org/10.5281/zenodo.15846963

<!-- END GENERATED METADATA -->
