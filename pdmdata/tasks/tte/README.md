# Time-to-event (TTE)

Time-to-event prediction estimates when a failure, alarm, or other
maintenance-relevant event will occur. This package owns the general TTE column
vocabulary (`time_to_event`, `event_observed`), entity-split helpers, and the
C-MAPSS TTE experiment bundle.

Exported `ENTITY_COLUMN` / `TIME_COLUMN` (`"entity"` / `"time"`) are the
generic vocabulary defaults for future adapters. The C-MAPSS bundle keeps
native names (`unit_number`, `cycle`) via `CMAPSS_ENTITY_COLUMN` /
`CMAPSS_TIME_COLUMN` in `pdmdata.tasks.cmapss_common`; do not assume frames
use `"entity"` or `"time"`.

RUL is a TTE profile (event = end of useful life) with a NASA-oriented `RUL`
column name; see the [RUL task](../rul/README.md) for that view.

Canonical taxonomy id: `time_to_event`.

## Quick start (C-MAPSS)

```python
import pdmdata
from pdmdata.tasks.tte import eval_cmapss, prepare_cmapss

pdmdata.download("cmapss")
bundle = prepare_cmapss(
    "FD001",
    time_cap=125,
    validation_fraction=0.2,
    drop_constant_features=True,
)
X, y, groups = bundle.train.to_numpy()
entities = bundle.entity_table()
metrics = eval_cmapss(bundle, predictions)
```

## Other datasets

```python
from pdmdata.tasks.tte import (
    prepare_backblaze,
    prepare_care,
    prepare_mapm,
    prepare_ncmapss,
    prepare_scania_x,
    prepare_xjtu_sy,
)

ncmapss = prepare_ncmapss("ds01", time_cap=125, drop_constant_features=True)
xjtu = prepare_xjtu_sy(
    condition="35Hz12kN",
    test_fraction=0.25,
    validation_fraction=0.25,
    time_cap=100,
)
scania = prepare_scania_x(test_fraction=0.2, time_cap=48)
drives = prepare_backblaze("2025-q1", test_fraction=0.2)
machines = prepare_mapm(test_fraction=0.2)
turbines = prepare_care("A", validation_fraction=0.2)
```

- **N-CMAPSS**: development units are observed failures; HDF5 `test` units are
  right-censored in `entity_table()`; last-cycle `Y_test` remaining time is
  used in `test_eval_split()` / `eval_ncmapss`.
- **XJTU-SY**: all bearings fail; held-out bearings form feature-only test
  rows; `test_eval_split()` / `eval_xjtu_sy` restores
  full-trajectory `time_to_event`.
- **Scania Component X**: training vehicles carry `time_to_event` and
  `event_observed` from the repair table. Censored rows stop at the end of
  observation. Official validation and test files only have time-window
  classes, so the bundle holds out training vehicles instead.
- **Backblaze**: `failure == 1` is the event day. Drives with no failure in
  the loaded quarter are right-censored at their last snapshot. Duration is
  whole days.
- **MAPM**: the event is the next component failure. Hours after the last
  failure, or the whole series when a machine never fails, are censored at
  the last telemetry time. Maintenance does not reset the clock.
  `entity_table()` has one row per inter-failure spell.
- **CARE**: one wind farm at a time. Anomaly onset is `event_start`; rows
  after onset are omitted. Normal events are censored at the last timestamp.
  The feature-only test set is the provider `prediction` window. Remaining
  time is in hours.

## Bundle contents

| Member | Role |
|---|---|
| `train` / `validation` | Rows with `time_to_event` and `event_observed` |
| `test` | Feature rows only (no target column) |
| `test_end_time` / `test_time` | Held-out labels for evaluation |
| `test_eval_split()` | Last row for C-MAPSS and N-CMAPSS; every held-out row otherwise |
| `entity_table()` | Duration and censoring. MAPM has one row per failure spell |

For C-MAPSS and N-CMAPSS, `entity_table()` keeps test engines right-censored
at the last observed cycle (no official remaining-time offset). Official
remaining-time scoring stays on `test_eval_split()`.

## Shared helpers

Callers may import `split_entities` and `filter_entities` for entity-level
hold-outs (integer unit ids or string bearing ids).

## Sequence samples and NPZ handoffs

`CmapssSequenceSpec` separates reusable data construction settings from model
and optimization settings. `prepare_cmapss_sequences` returns NumPy split
arrays plus metadata; `export_cmapss_sequences` saves them in a new directory.
They load previously downloaded data and never download automatically.

```python
from pdmdata.tasks.tte import CmapssSequenceSpec, export_cmapss_sequences

spec = CmapssSequenceSpec(
    subset="FD001",
    feature_columns=["sensor_2", "sensor_3", "sensor_4", "sensor_7"],
    seq_len=30,
    rul_cap=125,
    validation_fraction=0.2,
    split_seed=0,
    train_stride=5,
    validation_stride=5,
)
manifest = export_cmapss_sequences("experiment/outputs/new-run/data", spec)
```

The same API accepts a dictionary with these fields. FD001 through FD004 are
supported by the existing loader; no model-specific sensor list is imposed.
Explicit `validation_units` freezes an inner hold-out. `evaluation_units`
reserves a separate outer scoring cohort from official training engines
before the inner split. These IDs must be disjoint. Pass
`include_test=False` to `prepare_cmapss_sequences` or
`export_cmapss_sequences` to avoid reading official test series or labels. Min-Max
parameters are fitted on **training entities after the split**, including
all their observations, and applied without clipping to held-out samples.
Constant training features use scale 1. Refitting within each outer training
fold is the caller's responsibility. No automatic feature selection is used.

Training and validation windows use the configured stride and always retain
an entity's final row when it meets the optional `minimum_endpoint` cycle
filter. The filter also applies to training-cohort evaluation engines,
while full normalized points remain available. Official test endpoints are
not filtered by this policy. Official test uses one last-observation window per
entity with the official scoring label. Short series repeat their first
observation and time on the left; the observation mask remains one. This
padding denotes artificial repetition, not measured earlier history.
`rul_cap=None` retains uncapped remaining-time labels.

The schema-1 NPZ arrays are `x`, `y`, `times`, `mask`, `entity`, and `endpoint`.
Times and targets use cycles; `x` and `y` are float32, entities are int64.
Training/validation/evaluation also retain `points_x`, `points_y`, `points_entity` for
complete-series derivations. No model may use future rows, final series
lengths, or target metadata as input. The manifest records split IDs, scaling,
data settings, raw/generated file hashes, installed package version, generator
source hashes, Python/library versions, and editable-checkout commit/dirty
state when available. Existing directories are never overwritten.

For common-endpoint window comparisons, derive from an existing handoff:

```python
from pdmdata.tasks.tte import export_derived_windows

manifest = export_derived_windows(
    "experiment/outputs/source-run/data",
    "experiment/outputs/new-run/data/window_60",
    length=60,
    minimum_endpoint=90,
)
```

This checks full point order, existing windows, contiguous cycles 1..n,
labels, split IDs, and file hashes. It preserves the source normalization
and existing endpoint order, reads only train/validation, and rejects short
padded source windows. It is a completed C-MAPSS cohort helper, not an
irregular-time or censored-test reconstructor. The manifest keeps original
raw-data provenance and adds a separate `sample_generator` provenance block.
`build_sequence_samples`, `complete_sequences`, and `common_windows` expose
the lower-level operations for custom task adapters. PyTorch batches, learning
samplers, teacher predictions, and learned features belong to model code.

## Related

- RUL profile: [RUL task](../rul/README.md)
- Shared helpers: `pdmdata.tasks.cmapss_common`, `pdmdata.tasks.metrics`
- Usage notes: [usage guide](../../../docs/usage.md#tte-task-view-c-mapss)
- Notebook: [time-to-event-prediction](../../../notebooks/tasks/time-to-event-prediction.ipynb)
- Dataset: [C-MAPSS](../../datasets/cmapss/README.md)

## Citation

```bibtex
@inproceedings{nakamura2026fast,
  title={Fast Mining and Dynamic Time-to-Event Prediction
         over Multi-sensor Data Streams},
  author={Nakamura, Kota and Kawabata, Koki and
          Matsubara, Yasuko and Sakurai, Yasushi},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference
             on Knowledge Discovery and Data Mining V. 1},
  pages={1089--1100},
  year={2026}
}
```


## Explicit PC-MambaSDE-style sequence profile

`CmapssSequenceSpec` additionally supports `normalization="train_standard"`,
`smoothing_alpha=0.8` (causal normalized EMA within each engine), and
`short_test_policy="linear_resample"` (fit a line to the observed short history
then resample it to the requested window length). Defaults preserve previous
Min-Max, no-smoothing, first-value-padding arrays and metadata. Scaling uses
training engines only; no test statistics are fitted.

`HigsSpec` and `higs_corrupt` implement a versioned public-code-style benchmark
transform: independent channel dropout, per-step burst starts, Poisson lengths,
whole- or half-channel bursts, noise, and sorted/clipped jittered local times.
Use a fixed NumPy Generator to give competing models the same corrupted inputs.
The code-level burst/time rules differ from the paper's Poisson-count/Gaussian-
length and iterative monotonic-jitter equations. This is not an assertion of
complete paper-result reproduction.

Positive `noise_rul_factor` explicitly uses RUL in cycle units to simulate
label-dependent benchmark noise; it is not an unlabeled deployment transform.
Set it to zero for label-free noise. No labels are returned as model features.
Samples with entirely missing observations are retained. These NumPy-only APIs
never import research models or PyTorch. Reference: PC-MambaSDE public commit
`6d4b653cb83d40828b8e7abc7e2faaf291053628`, `utils/irregularity.py`.
