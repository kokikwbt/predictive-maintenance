# Genesis Demonstrator Data for Machine Learning

<!-- BEGIN GENERATED METADATA -->

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
| Feature dimensions | 18 features in labeled files; 23 features in the other files |
| Feature counting | Genesis_AnomalyLabels.csv and Genesis_StateMachineLabel.csv have 20 columns including Timestamp and Label. Genesis_lineardrive.csv, Genesis_normal.csv, and Genesis_pressure.csv have 24 columns including Timestamp. Counts include continuous signals, setpoints, and discrete machine states. |
| Feature-count source | [inIT-OWL on Kaggle](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning) |
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
| `Timestamp` | Float64 | timestamp | Observation time recorded as Unix seconds; the state recording contains two backward clock steps. Preserve source order when inspecting transitions. |
| `MotorData.SetCurrent / ActCurrent` | Float64 | sensor | Motor-current setpoint and measured value. |
| `MotorData.SetPosition / ActPosition` | Float64 | sensor | Motor-position setpoint and measured value. |
| `MotorData.SetSpeed / ActSpeed` | Float64 | sensor | Motor-speed setpoint and measured value. |
| `MotorData.SetAcceleration / IsAcceleration` | Float64 | sensor | Acceleration setpoint and measured value. |
| `MotorData.SetForce / IsForce` | Float64 | sensor | Force setpoint and measured value. |
| `MotorData.*, PLC_PRG.*, NVL_*` | Boolean/Integer | machine-state | Discrete state signals for position, gripping, material, and storage mechanisms. |
| `Label` | Integer | target | File-dependent target: state-machine IDs 0–8 in Genesis_StateMachineLabel.csv; separate anomaly IDs 0–2 in Genesis_AnomalyLabels.csv. No verified mapping from state IDs to named actions is supplied here. |

## Usage notes

- Distribution depends on Kaggle.
- The column set and presence of the Label column vary by file.

## Download

```bash
uv run --locked python scripts/download.py gdd
```

Kaggle dataset: [`inIT-OWL/genesis-demonstrator-data-for-machine-learning`](https://www.kaggle.com/datasets/inIT-OWL/genesis-demonstrator-data-for-machine-learning)

## Suggested citation

Schuster, R., Moriz, N. and von Birgelen, A. (2018). Genesis demonstrator data for machine learning.

<!-- END GENERATED METADATA -->

## Explore machine states

The [GDD notebook](../../../notebooks/datasets/gdd.ipynb) contains executed real-data
plots, state frequencies, contiguous segment lengths, binary flag profiles, and
transition counts. State IDs are kept numeric because their action-name mapping
has not been verified. Machine-state labels and anomaly labels are distinct.

```python
import pdmdata
from pdmdata.datasets.gdd.preprocessing import state_segments
from pdmdata.datasets.gdd.viz import plot_states

frame = pdmdata.load("gdd", series="state")
segments = state_segments(frame)
figure = plot_states(frame, start=0, stop=800)
```

The plots preserve original sample order because the source timestamps contain
backward steps. Segment endpoints use half-open sample intervals, not physical
state-duration annotations.
