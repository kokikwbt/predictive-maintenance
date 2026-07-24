# Production Plant Data for Condition Monitoring

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

Condition-monitoring data from eight run-to-failure experiments on eight important production-plant components.

| Item | Details |
|---|---|
| ID | `ppd` |
| Name | Production Plant Data for Condition Monitoring |
| Provider | [inIT-OWL on Kaggle](https://www.kaggle.com/datasets/inIT-OWL/production-plant-data-for-condition-monitoring) |
| DOI | None |
| Availability | available (checked: 2026-07-24) |
| Access | Kaggle |
| License | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Run-to-failure time series |
| Tasks | Degradation estimation, Anomaly detection, RUL research |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `Timestamp` | Integer | time | Time index indicating observation order within an experiment. |
| `L_1..L_10` | Float64 | sensor | Paired current- and speed-related signals for five components. |
| `A_1..A_5` | Float64 | sensor | Five measurements associated with the sixth component. |
| `B_1..B_5` | Float64 | sensor | Five measurements associated with the seventh component. |
| `C_1..C_5` | Float64 | sensor | Five measurements associated with the eighth component. |
| `trial/component` | Categorical | entity | Experiment and component identifier C7, C8, C9, C11, C13, C14, C15, or C16, represented by the file name. |

## Usage notes

- C7 and C13 each span two files separated by a short interruption.
- Failure time and RUL are not directly labeled and require interpretation based on experiment endpoints and expert assessments.

## Download

```bash
python scripts/download.py ppd
```

Kaggle dataset: [`inIT-OWL/production-plant-data-for-condition-monitoring`](https://www.kaggle.com/datasets/inIT-OWL/production-plant-data-for-condition-monitoring)

## Suggested citation

von Birgelen, A. et al. (2018). Self-Organizing Maps for Anomaly Localization and Predictive Maintenance in Cyber-Physical Production Systems.

<!-- END GENERATED METADATA -->
