# Remaining useful life (RUL)

RUL is a time-to-event profile where the event is end of useful life and the
target at each row is remaining operating cycles. Dataset loaders stay
source-faithful; this package builds a Polars-tabular experiment bundle for
modeling.

Canonical taxonomy id: `rul_prediction`.

## Quick start (C-MAPSS)

```python
import pdmdata
from pdmdata.tasks.rul import (
    evaluate_test_predictions,
    prepare_cmapss,
)

pdmdata.download("cmapss")
bundle = prepare_cmapss(
    "FD001",
    rul_cap=125,
    validation_fraction=0.2,
    drop_constant_features=True,
)
X, y, groups = bundle.train.to_numpy()
eval_split = bundle.test_eval_split()
metrics = evaluate_test_predictions(
    bundle, model.predict(eval_split.X().to_numpy())
)
```

## Other datasets

```python
from pdmdata.tasks.rul import prepare_ncmapss, prepare_xjtu_sy

ncmapss = prepare_ncmapss("ds01", rul_cap=125, drop_constant_features=True)
xjtu = prepare_xjtu_sy(
    condition="35Hz12kN",
    test_fraction=0.25,
    validation_fraction=0.25,
    rul_cap=100,
)
```

- **N-CMAPSS** (`prepare_ncmapss`): development units for train/validation;
  HDF5 `test` units for feature-only rows; last-cycle `Y_test` RUL for
  `test_eval_split()` / `evaluate_ncmapss_test_predictions`.
- **XJTU-SY** (`prepare_xjtu_sy`): run-to-failure bearings held out as test;
  `test_eval_split()` scores every snapshot (not only the final `RUL=0`).

## Bundle contents

| Member | Role |
|---|---|
| `train` / `validation` | Labeled rows with `RUL`, `time_to_event`, `event_observed` |
| `test` | Feature rows only (no target column) |
| `test_end_rul` / `test_rul` | Held-out labels for evaluation |
| `test_eval_split()` | Scoring view (last unit row for C-MAPSS/N-CMAPSS; all rows for XJTU-SY) |

Optional `rul_cap` applies piecewise `min(RUL, cap)` after constructing
uncapped remaining life. Validation entities are held out at the unit or
bearing level.

## Related

- General TTE formulation: [TTE task](../tte/README.md)
- Shared helpers: `pdmdata.tasks.cmapss_common`, `pdmdata.tasks.metrics`
- Usage notes: [usage guide](../../../docs/usage.md#rul-task-view-c-mapss)
- Notebook: [remaining-useful-life-prediction](../../../notebooks/tasks/remaining-useful-life-prediction.ipynb)
- Datasets: [C-MAPSS](../../datasets/cmapss/README.md),
  [N-CMAPSS](../../datasets/ncmapss/README.md),
  [XJTU-SY](../../datasets/xjtu_sy/README.md)
