# Alarm Logs in Packaging Industry

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

A sequence of timestamped alarm events collected from 20 industrial packaging machines.

| Item | Details |
|---|---|
| ID | `alpi` |
| Name | Alarm Logs in Packaging Industry |
| Provider | [Mendeley Data](https://data.mendeley.com/datasets/4nhx2x67cd) |
| DOI | [10.17632/4nhx2x67cd.1](https://doi.org/10.17632/4nhx2x67cd.1) |
| Availability | available (checked: 2026-07-24) |
| Access | Direct download |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Event sequence |
| Tasks | Alarm forecasting, Anomaly detection, Sequence forecasting |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Time-to-event prediction | Direct |
| Survival analysis | Requires target derivation |
| Event or sequence forecasting | Direct |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `timestamp` | Datetime | timestamp | Date and time at which the alarm was recorded. |
| `alarm` | UInt16 | event | Alarm code with 154 distinct values. |
| `serial` | UInt16 | entity | Anonymized identifier of the machine that generated the alarm. |

## Usage notes

- The collection period is 2019-02-21 through 2020-06-17.
- The alarm-code distribution is highly imbalanced.
- Cross-machine evaluation requires group-aware splitting to prevent leakage from the same machine.

## Suggested citation

Dalle Pezze, D. et al. (2021), “ALARM LOGS IN PACKAGING INDUSTRY (ALPI)”, Mendeley Data, V1.

<!-- END GENERATED METADATA -->
