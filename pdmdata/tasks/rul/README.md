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

## Bundle contents

| Member | Role |
|---|---|
| `train` / `validation` | Labeled rows with `RUL`, `time_to_event`, `event_observed` |
| `test` | Feature rows only (no target column) |
| `test_end_rul` | Official one-value-per-engine end-of-trajectory RUL |
| `test_eval_split()` | Last test row per unit joined with `test_end_rul` |

Optional `rul_cap` applies piecewise `min(RUL, cap)` after constructing
uncapped remaining life. Validation engines are held out at the unit level.

## Related

- General TTE formulation: [TTE task](../tte/README.md)
- Shared helpers: `pdmdata.tasks.cmapss_common`, `pdmdata.tasks.metrics`
- Usage notes: [usage guide](../../../docs/usage.md#rul-task-view-c-mapss)
- Notebook: [remaining-useful-life-prediction](../../../notebooks/tasks/remaining-useful-life-prediction.ipynb)
- Dataset: [C-MAPSS](../../datasets/cmapss/README.md)
