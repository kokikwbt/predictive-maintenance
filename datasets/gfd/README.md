# Gearbox Fault Diagnosis

<!-- BEGIN GENERATED METADATA -->

> This section is generated from `metadata.json`. Do not edit it directly.

## Dataset overview

Vibration signals recorded in four directions from a two-stage gearbox with healthy and broken-tooth gears under varying load.

| Item | Details |
|---|---|
| ID | `gfd` |
| Name | Gearbox Fault Diagnosis Data |
| Provider | [Open Energy Data Initiative](https://data.openei.org/submissions/623) |
| DOI | — |
| Availability | available (checked: 2026-07-24) |
| Access | Direct download |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Vibration time series |
| Tasks | Fault classification, Signal analysis |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Direct |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `sensor1..4` | Float64 | sensor | Measurements from vibration sensors mounted in four directions; the source CSV columns are a1 through a4. |
| `condition` | Categorical | target | Healthy (h) or broken-tooth (b) condition encoded in the file name. |
| `load` | UInt8 | operating-condition | Load from 0% through 90% in 10% increments, encoded in the file name. |

## Usage notes

- The dataset contains 20 files: two conditions at ten load levels.
- Condition and load are encoded in file names rather than CSV columns.

## Download

```bash
python scripts/download.py gfd
```

Source archive: [gearbox-fault-diagnosis.zip](https://data.openei.org/files/623/gearboxdata.zip)

## Suggested citation

Pandya, Y. and Parey, A. Gearbox Fault Diagnosis Data. OpenEI submission 623.

<!-- END GENERATED METADATA -->


Gearbox Fault Diagnosis Data set include the vibration dataset recorded by using SpectraQuest's Gearbox Fault Diagnostics Simulator.
- recorded with the help of 4 vibration sensors placed in four different direction.
- recorded under variation of load from '0' to '90' percent.
- recorded in two different scenario:
    1) Healthy condition
    2) Broken Tooth Condition

## References
- [https://www.kaggle.com/brjapon/gearbox-fault-diagnosis](https://www.kaggle.com/brjapon/gearbox-fault-diagnosis)
- [https://openei.org/datasets/dataset/gearbox-fault-diagnosis-data](https://openei.org/datasets/dataset/gearbox-fault-diagnosis-data)
