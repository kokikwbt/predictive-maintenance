# One Year Industrial Component Degradation

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

High-frequency machine measurements tracking cutting-blade degradation in an industrial shrink-wrapper over twelve months.

| Item | Details |
|---|---|
| ID | `oyicd` |
| Name | One Year Industrial Component Degradation |
| Provider | [inIT-OWL on Kaggle](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation) |
| DOI | — |
| Availability | available (checked: 2026-07-24) |
| Access | Kaggle |
| License | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Multivariate degradation time series |
| Tasks | Degradation estimation, Anomaly detection, Operating-mode classification, RUL research |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Operating-state classification | Direct |
| Condition estimation | Requires target derivation |
| Remaining useful life prediction | Requires target derivation |
| Time-to-event prediction | Requires target derivation |
| Survival analysis | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `timestamp` | Float64 | time | Time within an approximately eight-second recording sampled at four-millisecond intervals. |
| `pCut::*` | Float64 | sensor | Cutting-assembly measurements, including motor and blade-related signals. |
| `filename` | String | entity | Recording identifier encoding month, day, start time, sample number, and operating mode. |
| `mode` | UInt8 | operating-condition | Operating mode from 1 through 8 encoded in the source filename. |
| `month` | UInt8 | degradation-time | Relative month from 1 through 12 encoded in the source filename. |

## Usage notes

- The distribution contains 518 CSV recordings; each recording normally contains 2,048 samples.
- Source filenames follow MM-DDTHHMMSS_NUM_modeX.csv, where MM is a relative month rather than a calendar month.
- Component replacement, failure time, and RUL are not explicitly labeled and require interpretation from degradation trends.
- Download and any required authentication are handled by the official Kaggle CLI.

## Download

```bash
python scripts/download.py oyicd
```

Kaggle dataset: [`inIT-OWL/one-year-industrial-component-degradation`](https://www.kaggle.com/datasets/inIT-OWL/one-year-industrial-component-degradation)

## Suggested citation

von Birgelen, A., Buratti, D., Mager, J. and Niggemann, O. (2018). Self-Organizing Maps for Anomaly Localization and Predictive Maintenance in Cyber-Physical Production Systems.

<!-- END GENERATED METADATA -->
