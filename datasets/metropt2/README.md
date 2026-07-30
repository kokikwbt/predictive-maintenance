# MetroPT2: A Benchmark Dataset for Predictive Maintenance

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

A 1 Hz multivariate time series from a metro-train air production unit, with analogue and digital sensor signals and documented air- and oil-leak intervals.

| Item | Details |
|---|---|
| ID | `metropt2` |
| Name | MetroPT2: A Benchmark Dataset for Predictive Maintenance |
| Provider | [INESC TEC on Zenodo](https://zenodo.org/records/7766691) |
| DOI | [10.5281/zenodo.7766691](https://doi.org/10.5281/zenodo.7766691) |
| Availability | available (checked: 2026-07-25) |
| Access | Direct download from Zenodo |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Large multivariate equipment time series |
| Tasks | Online anomaly detection, Failure prediction, RUL research |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Requires target derivation |
| Condition estimation | Requires target derivation |
| Remaining useful life prediction | Requires target derivation |
| Time-to-event prediction | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `timestamp` | Datetime | timestamp | Observation time for measurements logged at 1 Hz. |
| `analogue sensors` | Float64 | sensor | Pressure, motor-current, oil-temperature, flowmeter, GPS, and related analogue measurements. |
| `digital signals` | Boolean/Integer | machine-state | Control and discrete signals including air-intake-valve states. |
| `failure interval` | Datetime interval | derived-target | Air-leak and oil-leak intervals documented by the provider outside the CSV labels. |

## Usage notes

- The current version contains about 7.1 million observations and 21 attributes.
- The source CSV is unlabeled; task targets must be derived from the two provider-documented failure intervals.
- The 1.2 GB CSV is opt-in and is not downloaded by scripts/bootstrap.sh.

## Download

```bash
python scripts/download.py metropt2
```

Source archive: [MetroPT2.csv](https://zenodo.org/records/7766691/files/MetroPT2.csv?download=1)

## Suggested citation

Veloso, B., Gama, J., Ribeiro, R. and Pereira, P. (2023). MetroPT2: A Benchmark dataset for predictive maintenance. Zenodo. https://doi.org/10.5281/zenodo.7766691

<!-- END GENERATED METADATA -->
