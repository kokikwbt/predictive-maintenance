# Microsoft Azure Predictive Maintenance

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

Synthetic data linking 2015 sensor telemetry, errors, maintenance, failures, and machine metadata for 100 simulated machines.

| Item | Details |
|---|---|
| ID | `mapm` |
| Name | Microsoft Azure Predictive Maintenance |
| Provider | [Microsoft Azure sample / Kaggle mirror](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance) |
| DOI | None |
| Availability | available (checked: 2026-07-24) |
| Access | Kaggle CLI |
| License | Unknown |
| Commercial use | Unknown |
| Redistribution | Unknown |
| Data type | Multi-table time series |
| Tasks | Failure prediction, RUL estimation, Maintenance analysis |

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
- The former Microsoft Azure Blob URLs were unavailable when checked on 2026-07-24; automated retrieval uses the Kaggle mirror.
- Treat this as synthetic data.
- No dataset-specific license was found, so redistribution and commercial-use rights cannot be asserted.

## Download

```bash
python scripts/download.py mapm
```

Kaggle dataset: [`arnabbiswas1/microsoft-azure-predictive-maintenance`](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

## Suggested citation

Microsoft Azure Predictive Maintenance sample dataset.

<!-- END GENERATED METADATA -->
