# Time-to-event (TTE)

Time-to-event prediction estimates when a failure, alarm, or other
maintenance-relevant event will occur. This package owns the general TTE column
vocabulary (`time_to_event`, `event_observed`), entity-split helpers, and the
C-MAPSS TTE experiment bundle.

RUL is a TTE profile (event = end of useful life) with a NASA-oriented `RUL`
column name; see the [RUL task](../rul/README.md) for that view.

Canonical taxonomy id: `time_to_event`.

## Quick start (C-MAPSS)

```python
import pdmdata
from pdmdata.tasks.tte import (
    evaluate_test_predictions,
    prepare_cmapss,
)

pdmdata.download("cmapss")
bundle = prepare_cmapss(
    "FD001",
    time_cap=125,
    validation_fraction=0.2,
    drop_constant_features=True,
)
X, y, groups = bundle.train.to_numpy()
entities = bundle.entity_table()
metrics = evaluate_test_predictions(bundle, predictions)
```

## Bundle contents

| Member | Role |
|---|---|
| `train` / `validation` | Rows with `time_to_event` and `event_observed=True` |
| `test` | Feature rows only (no target column) |
| `test_end_time` | Official remaining time per test engine |
| `test_eval_split()` | Last test row per unit with joined evaluation time |
| `entity_table()` | One row per engine with duration and censoring |

`entity_table()` keeps NASA test engines right-censored at the last observed
cycle (no official RUL offset). Official remaining-time scoring stays on
`test_eval_split()`.

## Shared helpers

Callers may import `split_entities` and `filter_entities` for unit-level
hold-outs on other datasets that already expose entity IDs.

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
