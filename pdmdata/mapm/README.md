# Microsoft Azure Predictive Maintenance

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

Synthetic data linking 2015 sensor telemetry, errors, maintenance, failures, and machine metadata for 100 simulated machines.

| Item | Details |
|---|---|
| ID | `mapm` |
| Name | Microsoft Azure Predictive Maintenance |
| Provider | [Microsoft sqlworkshops](https://github.com/microsoft/sqlworkshops/tree/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data) |
| DOI | — |
| Availability | available (checked: 2026-09-06) |
| Access | Direct download from Microsoft GitHub |
| License | Unknown |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Multi-table time series |
| Feature dimensions | 4 telemetry features |
| Feature counting | Counts volt, rotate, pressure, and vibration; datetime and machineID are excluded from the 6 telemetry columns. Joining machine model and age adds two input fields before categorical encoding. Error, maintenance, and failure tables require task-specific joins or aggregation, so the engineered dimension is not fixed. |
| Feature-count source | [Microsoft sqlworkshops](https://github.com/microsoft/sqlworkshops/tree/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data) |
| Tasks | Failure prediction, RUL estimation, Maintenance analysis |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Direct |
| Remaining useful life prediction | Requires target derivation |
| Time-to-event prediction | Direct |
| Survival analysis | Direct |
| Event or sequence forecasting | Direct |
| Maintenance-policy evaluation | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `datetime` | Datetime | timestamp | Timestamp for telemetry, error, maintenance, and failure events. |
| `machineID` | UInt16 | entity | Machine identifier from 1 through 100. |
| `volt` | Float64 | sensor | Hourly mean voltage. |
| `rotate` | Float64 | sensor | Hourly mean rotation. |
| `pressure` | Float64 | sensor | Hourly mean pressure. |
| `vibration` | Float64 | sensor | Hourly mean vibration. |
| `errorID` | Categorical | event | Type of non-fatal machine error. |
| `comp` | Categorical | maintenance-event | Component replaced during maintenance or after failure. |
| `model` | Categorical | asset-attribute | Machine model. |
| `age` | UInt8 | asset-attribute | Machine age. |

## Usage notes

- The original Azure AI Notebook was retired in 2020.
- The former Azure Blob host did not resolve on 2026-09-06. Downloads use the five source CSVs in Microsoft sqlworkshops on GitHub and bundle their unchanged contents locally. Kaggle is not required.
- Treat this as synthetic data.
- No dataset-specific license was found, so redistribution and commercial-use rights cannot be asserted.

## Download

```bash
uv run --locked python scripts/download.py mapm
```

Source files (bundled locally without changing their contents):

- [PdM_telemetry.csv](https://raw.githubusercontent.com/microsoft/sqlworkshops/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data/PdM_telemetry.csv)
- [PdM_errors.csv](https://raw.githubusercontent.com/microsoft/sqlworkshops/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data/PdM_errors.csv)
- [PdM_maint.csv](https://raw.githubusercontent.com/microsoft/sqlworkshops/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data/PdM_maint.csv)
- [PdM_failures.csv](https://raw.githubusercontent.com/microsoft/sqlworkshops/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data/PdM_failures.csv)
- [PdM_machines.csv](https://raw.githubusercontent.com/microsoft/sqlworkshops/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data/PdM_machines.csv)

## Suggested citation

Microsoft Azure Predictive Maintenance sample dataset.

<!-- END GENERATED METADATA -->
