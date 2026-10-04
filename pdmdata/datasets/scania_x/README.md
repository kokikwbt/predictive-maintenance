# SCANIA Component X: Truck Fleet Readouts and Repair Records

<!-- BEGIN GENERATED METADATA -->

## Dataset overview

Real-world operational readouts, repair records, and specifications for an anonymized engine component across more than 33,000 SCANIA heavy-duty trucks, released for the IDA 2024 Industrial Challenge.

| Item | Details |
|---|---|
| ID | `scania_x` |
| Name | SCANIA Component X Dataset |
| Provider | [Swedish National Data Service (SND) / Researchdata.se](https://researchdata.se/en/catalogue/dataset/2024-34) |
| DOI | [10.5878/bnh5-ka77](https://doi.org/10.5878/bnh5-ka77) |
| Availability | available (checked: 2026-10-04) |
| Access | Direct per-file CSV downloads from Researchdata.se (~1.65 GB; no account required) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Commercial use | Yes |
| Redistribution | Yes |
| Data type | Fleet-scale irregular multivariate time series with repair records |
| Feature dimensions | 105 operational features from 14 anonymized variables, plus 8 categorical specifications |
| Feature counting | Readout files have 107 columns: vehicle_id, time_step, 8 numerical counters (171_0, 666_0, 427_0, 837_0, 309_0, 835_0, 370_0, 100_0), and 6 histograms with 10 (167), 10 (272), 11 (291), 10 (158), 20 (459), and 36 (397) bins. Specification files add Spec_0 to Spec_7 per vehicle. |
| Feature-count source | [Kharazian et al., Scientific Data (2025)](https://doi.org/10.1038/s41597-025-04802-6) |
| Tasks | Imminent-failure classification, Time-to-event prediction, Survival analysis, Cost-sensitive maintenance decisions |

## Experiment-task suitability

| Task | Support |
|---|---|
| Anomaly detection | Requires target derivation |
| Fault or health-state classification | Direct |
| Condition estimation | Requires target derivation |
| Remaining useful life prediction | Requires target derivation |
| Time-to-event prediction | Direct |
| Survival analysis | Direct |
| Maintenance-policy evaluation | Requires target derivation |

## Attributes

| Attribute or group | Type | Role | Description |
|---|---|---|---|
| `vehicle_id` | Int64 | entity | Anonymized truck identifier; train, validation, and test vehicles are disjoint. |
| `time_step` | Float64 | time | Anonymized operating time since Component X started working; readouts are irregular per vehicle. |
| `171_0, 666_0, 427_0, 837_0, 309_0, 835_0, 370_0, 100_0` | Float64 | sensor | Mostly cumulative numerical counters. |
| `167_*, 272_*, 291_*, 158_*, 459_*, 397_*` | Float64 | sensor | Histogram bins named variableid_binindex (10, 10, 11, 10, 20, and 36 bins); outer bins are open-ended. |
| `Spec_0 … Spec_7` | Utf8 | asset-attribute | Anonymized categorical truck specifications (Cat0, Cat1, …). |
| `length_of_study_time_step` | Float64 | target | Train only: time of the first Component X repair, or end of observation when censored. |
| `in_study_repair` | UInt8 | event | Train only: 1 if Component X was repaired at length_of_study_time_step, 0 if right-censored. |
| `class_label` | UInt8 | target | Validation/test only: time window of the last readout before repair: 0 (>48), 1 (48–24), 2 (24–12), 3 (12–6), 4 (6–0) time steps. |

## Usage notes

- Train readouts cover each vehicle's full study period; validation and test readouts are truncated at a randomly selected last readout, and labels refer to that last readout.
- About 9.6% of training vehicles (2,272 of 23,550) have a repair event; the rest are right-censored. Validation and test labels are strongly skewed toward class 0.
- Timestamps, readout frequencies, and repair frequencies were anonymized and may not reflect actual truck usage. Variable names are anonymized IDs.
- Fewer than 1% of values per readout feature are missing; loaders keep them as null.
- Downloads use the nine CSVs of version 3 (DOI 10.5878/bnh5-ka77), verify each SND-published SHA-256, and bundle them locally. Data files are identical to version 2 (DOI 10.5878/jvb5-d390), which the Scientific Data paper cites.
- Expect about 3.3 GB on disk after extraction (local bundle plus extracted CSVs).

## Download

```bash
uv run --locked python scripts/download.py scania_x
```

Source files (bundled locally without changing their contents):

- [train_operational_readouts.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/train_operational_readouts.csv)
- [train_specifications.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/train_specifications.csv)
- [train_tte.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/train_tte.csv)
- [validation_operational_readouts.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/validation_operational_readouts.csv)
- [validation_specifications.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/validation_specifications.csv)
- [validation_labels.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/validation_labels.csv)
- [test_operational_readouts.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/test_operational_readouts.csv)
- [test_specifications.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/test_specifications.csv)
- [test_labels.csv](https://api.researchdata.se/dataset/2024-34/3/file/data/test_labels.csv)

## Suggested citation

Z. Kharazian, T. Lindgren, S. Magnússon, O. Steinert and O. Andersson Reyna (2025). SCANIA Component X dataset: a real-world multivariate time series dataset for predictive maintenance. Scientific Data, 12, 493. https://doi.org/10.1038/s41597-025-04802-6

<!-- END GENERATED METADATA -->


## Load the data

Download the nine source CSVs once (~1.65 GB, no account required). Each
file is checked against the SHA-256 published by SND before it is bundled.

```python
import pdmdata
from pdmdata.datasets.scania_x import (
    challenge_cost,
    last_readouts,
    load,
    vehicles,
)

pdmdata.download("scania_x")

# Operational readouts are lazy Polars scans; collect after selecting.
readouts = load("train", vehicle_id=2).collect()
# One row per vehicle: specifications plus repair record or class label.
train_vehicles = vehicles("train")
# Train readouts with derived time_to_event, event_observed, class_label.
labeled = load("train", with_labels=True).collect()
# Prediction points of the IDA 2024 challenge, with class_label attached.
validation = last_readouts("validation")
challenge_cost(validation["class_label"], [0] * validation.height)
```

| Table | Splits | Content |
|---|---|---|
| `readouts` | train, validation, test | `vehicle_id`, `time_step`, 105 counter and histogram features |
| `specifications` | train, validation, test | `Spec_0` … `Spec_7` per vehicle |
| `tte` | train | `length_of_study_time_step`, `in_study_repair` |
| `labels` | validation, test | `class_label` of the last readout |

### Labels and censoring

Training readouts cover each vehicle's full study period. In `train_tte.csv`,
`in_study_repair = 1` marks the first repair of Component X at
`length_of_study_time_step`; `0` marks a right-censored vehicle. Validation
and test readouts end at a randomly selected readout, and `class_label`
gives the time window of that readout before repair:

| Class | Time before repair (time steps) |
|---|---|
| 0 | more than 48 (or no repair) |
| 1 | 48 to 24 |
| 2 | 24 to 12 |
| 3 | 12 to 6 |
| 4 | 6 to 0 |

`add_tte_targets` (used by `with_labels=True`) applies the same windows to
training readouts. The provider does not state boundary inclusion, so the
windows are half-open: class 4 is `[0, 6)`, class 1 is `[24, 48)`, and so on.
Censored readouts within 48 time steps of the end of observation receive a
null label because their class is unknown.

### Challenge cost

`challenge_cost(y_true, y_pred)` sums the expert-defined cost matrix of the
IDA 2024 Industrial Challenge (`COST_MATRIX[actual][predicted]`). Predicting
too late or missing a repair costs 200–500; an early or unnecessary alarm
costs 7–10.

### Visualization

```python
figure = pdmdata.visualize(
    "scania_x", "counter_trajectory", load("train"), entity=2
)
```

`viz.plot_histogram_evolution(frame, variable="459")` draws the bins of one
histogram variable across the readouts of one collected vehicle.

Files are stored under `data/raw/scania_x/` by default.

## References

- Dataset landing page (version 3, DOI
  [10.5878/bnh5-ka77](https://doi.org/10.5878/bnh5-ka77)):
  [Researchdata.se catalogue entry 2024-34](https://researchdata.se/en/catalogue/dataset/2024-34)
- Version 2, cited by the data descriptor (identical data files):
  [10.5878/jvb5-d390](https://doi.org/10.5878/jvb5-d390)
- Data descriptor: Z. Kharazian, T. Lindgren, S. Magnússon, O. Steinert and
  O. Andersson Reyna, "SCANIA Component X dataset: a real-world multivariate
  time series dataset for predictive maintenance," *Scientific Data* 12, 493
  (2025). [doi:10.1038/s41597-025-04802-6](https://doi.org/10.1038/s41597-025-04802-6)
- Provider documentation PDF (final article version):
  [Scania_Component_X.pdf](https://api.researchdata.se/dataset/2024-34/3/file/documentation/Scania_Component_X.pdf)
- IDA 2024 Industrial Challenge description, including the label windows
  and cost matrix:
  [2024_IDA_challenge_v2.pdf](https://api.researchdata.se/dataset/2024-34/3/file/documentation/2024_IDA_challenge_v2.pdf)
- Machine-readable file list with SHA-256 checksums:
  [Researchdata.se API record](https://api.researchdata.se/dataset/2024-34/3)
- Earlier SCANIA benchmark from the IDA 2016 challenge (no temporal
  information): [APS Failure at Scania Trucks, UCI](https://doi.org/10.24432/C51S51)
