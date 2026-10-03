# Predictive-maintenance tasks

This directory holds the task taxonomy and experiment views. Dataset loaders
stay source-faithful; task packages build Polars-tabular bundles (targets,
entity splits, caps, and model-ready `X` / `y` / `groups`).

Compare which datasets support each task in the
[dataset catalog](../README.md#predictive-maintenance-tasks).

## Task list

Time-to-event prediction is the broad task of predicting when an event will
occur. Survival analysis is a time-to-event approach designed to model
censored observations, so it is listed separately when a dataset can support
that experimental design. RUL is a TTE profile where the event is end of
useful life.

| Task | Id | Status | Guide |
|---|---|---|---|
| Anomaly detection | `anomaly_detection` | Taxonomy only | — |
| Fault or health-state classification | `fault_classification` | Taxonomy only | — |
| Operating-state classification | `operating_state_classification` | Taxonomy only | — |
| Condition estimation | `condition_estimation` | Taxonomy only | — |
| Remaining useful life prediction | `rul_prediction` | Package | [rul/](rul/README.md) |
| Time-to-event prediction | `time_to_event` | Package | [tte/](tte/README.md) |
| Survival analysis | `survival_analysis` | Taxonomy only | — |
| Event or sequence forecasting | `event_forecasting` | Taxonomy only | — |
| Maintenance-policy evaluation | `maintenance_policy` | Taxonomy only | — |

- **Package:** `pdmdata/tasks/<id>/` with `prepare.py` and a user-facing README.
- **Taxonomy only:** listed in `taxonomy.py` and the catalog matrix; no
  experiment package yet.

## Implemented packages

### [Remaining useful life (RUL)](rul/README.md)

Predict remaining cycles or time before a defined failure endpoint. For
C-MAPSS, `prepare_cmapss` builds labeled train/validation splits, feature-only
test rows, and the official last-cycle scoring view.

```python
from pdmdata.tasks.rul import prepare_cmapss, evaluate_test_predictions

bundle = prepare_cmapss("FD001", rul_cap=125)
```

### [Time-to-event (TTE)](tte/README.md)

Predict when a failure or other maintenance-relevant event will occur, with
explicit `time_to_event` / `event_observed` columns and an entity-level
censoring table.

```python
from pdmdata.tasks.tte import prepare_cmapss

bundle = prepare_cmapss("FD001", time_cap=125)
entities = bundle.entity_table()
```

## Shared modules

| Module | Role |
|---|---|
| `taxonomy.py` | Canonical task ids and support levels |
| `types.py` | `TabularSplit` container |
| `metrics.py` | Shared scoring helpers (MAE, RMSE, NASA score) |
| `cmapss_common.py` | C-MAPSS feature/split helpers used by RUL and TTE |

## Related

- [Dataset catalog](../README.md) — dataset × task matrix
- [Usage guide](../../docs/usage.md) — RUL / TTE API notes
- [Task notebooks](../../notebooks/tasks/README.md) — runnable examples
