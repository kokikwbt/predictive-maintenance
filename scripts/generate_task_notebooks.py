#!/usr/bin/env python3
"""Generate executable notebooks for supported predictive-maintenance tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "tasks"

SETUP = """from pathlib import Path
import sys

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import pdmdata"""


def markdown(source: str) -> Dict[str, object]:
    """Create a Markdown notebook cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> Dict[str, object]:
    """Create an empty-output code notebook cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def notebook(cells: List[Dict[str, object]]) -> Dict[str, object]:
    """Create a notebook document using the project kernel."""
    for index, cell in enumerate(cells):
        cell["id"] = "cell-{:02d}".format(index)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (pdmdata)",
                "language": "python",
                "name": "pdmdata",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


NOTEBOOKS = {
    "anomaly-detection.ipynb": notebook(
        [
            markdown(
                """# Anomaly detection with gearbox vibration

Anomaly detection asks whether a new observation differs from normal operation.
This example treats healthy GFD vibration windows as the reference distribution
and uses broken-tooth recordings only for evaluation.

## Learning goals

- build window-level vibration features with Polars;
- avoid training an anomaly detector on known faults;
- evaluate anomaly scores against held-out healthy and faulty windows."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score"""
            ),
            code(
                """pdmdata.download("gfd")

def vibration_windows(condition, load, window_size=256):
    frame = pdmdata.load("gfd", condition=condition, load=load)
    sensors = [column for column in frame.columns if column.startswith("sensor_")]
    return (
        frame.with_row_index("sample")
        .with_columns((pl.col("sample") // window_size).alias("window"))
        .group_by("window")
        .agg(
            *[pl.col(column).mean().alias(f"{column}_mean") for column in sensors],
            *[pl.col(column).std().alias(f"{column}_std") for column in sensors],
            *[
                (pl.col(column) ** 2).mean().sqrt().alias(f"{column}_rms")
                for column in sensors
            ],
        )
        .with_columns(
            pl.lit(condition).alias("condition"),
            pl.lit(load).alias("load"),
        )
    )

healthy = pl.concat([vibration_windows("healthy", load) for load in range(0, 100, 10)])
broken = pl.concat([vibration_windows("broken", load) for load in range(0, 100, 10)])
healthy.shape, broken.shape"""
            ),
            markdown(
                """## Train on normal operation

The split is made by load so that evaluation is not performed on the same
operating conditions used to fit the detector."""
            ),
            code(
                """feature_columns = [
    column for column in healthy.columns
    if column not in {"window", "condition", "load"}
]
train = healthy.filter(pl.col("load") < 70)
test = pl.concat([
    healthy.filter(pl.col("load") >= 70),
    broken.filter(pl.col("load") >= 70),
])

detector = IsolationForest(contamination="auto", random_state=0)
detector.fit(train.select(feature_columns).to_numpy())
anomaly_score = -detector.score_samples(test.select(feature_columns).to_numpy())
is_fault = (test.get_column("condition") == "broken").cast(pl.UInt8).to_numpy()
print(f"Window-level ROC AUC: {roc_auc_score(is_fault, anomaly_score):.3f}")"""
            ),
            code(
                """scored = test.select("condition", "load").with_columns(
    pl.Series("anomaly_score", anomaly_score)
)
for condition, color in [("healthy", "tab:blue"), ("broken", "tab:red")]:
    values = scored.filter(pl.col("condition") == condition)["anomaly_score"].to_numpy()
    plt.hist(values, bins=30, alpha=0.55, label=condition, color=color)
plt.xlabel("Isolation Forest anomaly score")
plt.ylabel("Windows")
plt.legend()
plt.show()"""
            ),
            markdown(
                """## Interpretation and limitations

A high score means dissimilarity from the healthy training windows, not a
diagnosed failure. Load is a confounder, so a serious experiment should compare
condition-aware normalization, multiple window sizes, and group-based
cross-validation."""
            ),
        ]
    ),
    "fault-classification.ipynb": notebook(
        [
            markdown(
                """# Fault classification with gearbox vibration

Fault classification predicts a discrete health state. GFD provides healthy and
broken-tooth recordings at ten loads, allowing supervised evaluation.

## Learning goals

- convert raw vibration into window-level features;
- split by operating load to test generalization;
- evaluate balanced accuracy and a confusion matrix."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score"""
            ),
            code(
                """pdmdata.download("gfd")

def window_features(condition, load, window_size=256):
    frame = pdmdata.load("gfd", condition=condition, load=load)
    sensors = [column for column in frame.columns if column.startswith("sensor_")]
    return (
        frame.with_row_index("sample")
        .with_columns((pl.col("sample") // window_size).alias("window"))
        .group_by("window")
        .agg(
            *[pl.col(column).mean().alias(f"{column}_mean") for column in sensors],
            *[pl.col(column).std().alias(f"{column}_std") for column in sensors],
            *[
                (pl.col(column) ** 2).mean().sqrt().alias(f"{column}_rms")
                for column in sensors
            ],
        )
        .with_columns(
            pl.lit(condition).alias("condition"),
            pl.lit(load).alias("load"),
        )
    )

features = pl.concat([
    window_features(condition, load)
    for condition in ("healthy", "broken")
    for load in range(0, 100, 10)
])
features.group_by("condition", "load").len().sort("condition", "load")"""
            ),
            code(
                """predictors = [
    column for column in features.columns
    if column not in {"window", "condition", "load"}
]
train = features.filter(pl.col("load") < 70)
test = features.filter(pl.col("load") >= 70)

classifier = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=0,
)
classifier.fit(train.select(predictors).to_numpy(), train["condition"].to_numpy())
prediction = classifier.predict(test.select(predictors).to_numpy())
print(
    "Held-out-load balanced accuracy:",
    f"{balanced_accuracy_score(test['condition'].to_numpy(), prediction):.3f}",
)"""
            ),
            code(
                """ConfusionMatrixDisplay.from_predictions(
    test["condition"].to_numpy(),
    prediction,
    labels=["healthy", "broken"],
)
plt.title("GFD held-out-load classification")
plt.show()"""
            ),
            markdown(
                """## Interpretation and limitations

Random row splitting would leak near-identical adjacent vibration samples.
Windowing and holding out complete loads is stricter, but results from one
gearbox should not be interpreted as cross-machine generalization."""
            ),
        ]
    ),
    "operating-state-classification.ipynb": notebook(
        [
            markdown(
                """# Operating-state classification with OYICD

Operating-state classification identifies how a machine is running rather than
whether it has failed. OYICD encodes one of eight modes in every recording name.

## Learning goals

- discover downloaded recordings without hard-coded paths;
- summarize each recording with Polars;
- classify modes while holding out later recordings."""
            ),
            code(
                SETUP
                + """
from pdmdata.datasets.oyicd import inventory, load
from pdmdata.datasets.oyicd.loader import SENSOR_COLUMNS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score"""
            ),
            code(
                """pdmdata.download("oyicd")
files = inventory().sort("filename")
print(f"Unique recordings: {files.height}")

def summarize_recording(filename):
    frame = load(filename)
    return frame.select(
        *[pl.col(column).mean().alias(f"{column}_mean") for column in SENSOR_COLUMNS],
        *[pl.col(column).std().alias(f"{column}_std") for column in SENSOR_COLUMNS],
    ).with_columns(
        pl.lit(filename).alias("recording"),
        pl.lit(frame["mode"][0]).alias("mode"),
    )

recordings = pl.concat([summarize_recording(name) for name in files["filename"]])
recordings.group_by("mode").len().sort("mode")"""
            ),
            code(
                """recordings = recordings.with_columns(
    pl.int_range(pl.len()).over("mode").alias("mode_order"),
    pl.len().over("mode").alias("mode_count"),
)
train = recordings.filter(pl.col("mode_order") < pl.col("mode_count") * 0.8)
test = recordings.filter(pl.col("mode_order") >= pl.col("mode_count") * 0.8)
predictors = [
    column for column, dtype in recordings.schema.items()
    if dtype.is_numeric() and column not in {"mode", "mode_order", "mode_count"}
]

classifier = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=0,
)
classifier.fit(train.select(predictors).fill_null(0).to_numpy(), train["mode"].to_numpy())
prediction = classifier.predict(test.select(predictors).fill_null(0).to_numpy())
print(
    "Later-recording balanced accuracy:",
    f"{balanced_accuracy_score(test['mode'].to_numpy(), prediction):.3f}",
)"""
            ),
            markdown(
                """## Interpretation and limitations

Mode is an operating condition, not a fault label. The dataset loader validates
and counts duplicate source copies once. Features use only the eight sensors;
time and mode columns are excluded. The split holds out later recordings within
each mode; it is not a single global time cutoff across all modes. A production model should also test whether mode recognition transfers
between machines and component ages."""
            ),
        ]
    ),
    "condition-estimation.ipynb": notebook(
        [
            markdown(
                """# Condition estimation with hydraulic sensor data

Estimate cycle-average cooling efficiency from pressure measurements in HydSys.
Each row represents one 60-second test cycle. The target is the continuous CE
virtual sensor, rather than the discrete cooler-condition labels in profile.txt.

## Learning goals

- summarize pressure waveforms into cycle-level features;
- align inputs and targets by cycle, despite different sampling rates;
- evaluate a regression baseline on later cycles."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error"""
            ),
            code(
                """pdmdata.download("hydsys")
pressure = pdmdata.load("hydsys", sensor="PS1").to_numpy()
efficiency = pdmdata.load("hydsys", sensor="CE").to_numpy()
assert pressure.shape[0] == efficiency.shape[0], "Cycle counts must match"
features = np.column_stack([
    pressure.mean(axis=1), pressure.std(axis=1),
    pressure.min(axis=1), pressure.max(axis=1),
])
target = efficiency.mean(axis=1)
features.shape, target.shape"""
            ),
            code(
                """split = int(len(target) * 0.8)
model = ExtraTreesRegressor(n_estimators=200, random_state=0, n_jobs=-1)
model.fit(features[:split], target[:split])
prediction = model.predict(features[split:])
actual = target[split:]
print(f"Cooling-efficiency MAE: {mean_absolute_error(actual, prediction):.3f}")"""
            ),
            code(
                """plt.scatter(actual, prediction, s=8, alpha=0.4)
limits = [min(actual.min(), prediction.min()), max(actual.max(), prediction.max())]
plt.plot(limits, limits, "--", color="black")
plt.xlabel("Actual cycle-average cooling efficiency (%)")
plt.ylabel("Predicted cooling efficiency (%)")
plt.show()"""
            ),
            markdown(
                """## Interpretation and limitations

This baseline estimates a contemporaneous virtual sensor from pressure; it does
not predict future failures or remaining useful life. CE is used only as a
target. The split preserves cycle order, but repeated test-rig conditions may
still occur in both partitions. It does not establish generalization to other
machines or unseen fault conditions."""
            ),
        ]
    ),
    "remaining-useful-life-prediction.ipynb": notebook(
        [
            markdown(
                """# Remaining useful life prediction with C-MAPSS

RUL is a time-to-event profile: the event is end of useful life and the target
is remaining operating cycles. `pdmdata.tasks.rul.prepare_cmapss` builds a
Polars-tabular experiment bundle from the source-faithful C-MAPSS loader.

## Learning goals

- prepare train / validation / test with unit-level hold-out;
- fit a row-level regressor without leaking across engines;
- score the official last-observation test protocol (MAE, RMSE, NASA score)."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import HistGradientBoostingRegressor
from pdmdata.tasks.rul import evaluate_test_predictions, prepare_cmapss"""
            ),
            code(
                """pdmdata.download("cmapss")
bundle = prepare_cmapss(
    "FD001",
    rul_cap=125,
    validation_fraction=0.2,
    random_state=0,
    drop_constant_features=True,
)
bundle.summary()"""
            ),
            code(
                """bundle.train.frame.select(
    "unit_number", "cycle", "RUL", "time_to_event", "event_observed"
).head()"""
            ),
            code(
                """model = HistGradientBoostingRegressor(
    loss="absolute_error", random_state=0
)
X_train, y_train, _ = bundle.train.to_numpy()
X_val, y_val, _ = bundle.validation.to_numpy()
model.fit(X_train, y_train)
validation_prediction = np.maximum(model.predict(X_val), 0)
from pdmdata.tasks.rul import regression_metrics
print(regression_metrics(y_val, validation_prediction))"""
            ),
            code(
                """X_all, y_all, _ = prepare_cmapss(
    "FD001",
    rul_cap=125,
    validation_fraction=0.0,
    drop_constant_features=True,
).train.to_numpy()
model.fit(X_all, y_all)
eval_split = bundle.test_eval_split()
prediction = np.maximum(model.predict(eval_split.X().to_numpy()), 0)
print(evaluate_test_predictions(bundle, prediction))"""
            ),
            markdown(
                """## Interpretation and limitations

`prepare_cmapss` keeps loaders source-faithful and applies RUL construction,
optional piecewise capping, constant-feature dropping, and unit hold-out in the
task layer. Uncapped RUL counts cycles to failure; it does not mark fault onset.
Report `rul_cap` and the NASA score because both change the task materially.
Test targets stay out of `bundle.test`; use `test_eval_split()` for the official
one-row-per-engine protocol."""
            ),
        ]
    ),
    "time-to-event-prediction.ipynb": notebook(
        [
            markdown(
                """# Time-to-event prediction with C-MAPSS

Time-to-event (TTE) estimates when a defined event will occur. Here the event
is engine failure and time is measured in operating cycles.
`pdmdata.tasks.tte.prepare_cmapss` builds a Polars-tabular bundle; RUL is the
same remaining-time quantity under an end-of-life event profile.

## Learning goals

- prepare row-level ``time_to_event`` with ``event_observed``;
- hold out engines for validation without leaking across units;
- contrast official remaining-time scoring with entity-level censoring."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import HistGradientBoostingRegressor
from pdmdata.tasks.tte import evaluate_test_predictions, prepare_cmapss
from pdmdata.tasks.metrics import regression_metrics"""
            ),
            code(
                """pdmdata.download("cmapss")
bundle = prepare_cmapss(
    "FD001",
    time_cap=125,
    validation_fraction=0.2,
    random_state=0,
    drop_constant_features=True,
)
bundle.summary()"""
            ),
            code(
                """bundle.train.frame.select(
    "unit_number", "cycle", "time_to_event", "event_observed"
).head()"""
            ),
            code(
                """plt.hist(
    bundle.train.frame["time_to_event"].to_numpy(), bins=60
)
plt.xlabel("Operating cycles until failure")
plt.ylabel("Observations")
plt.show()"""
            ),
            code(
                """model = HistGradientBoostingRegressor(
    loss="absolute_error", random_state=0
)
X_train, y_train, _ = bundle.train.to_numpy()
X_val, y_val, _ = bundle.validation.to_numpy()
model.fit(X_train, y_train)
print(regression_metrics(y_val, np.maximum(model.predict(X_val), 0)))"""
            ),
            code(
                """full = prepare_cmapss(
    "FD001",
    time_cap=125,
    validation_fraction=0.0,
    drop_constant_features=True,
)
model.fit(*full.train.to_numpy()[:2])
prediction = np.maximum(
    model.predict(bundle.test_eval_split().X().to_numpy()), 0
)
print(evaluate_test_predictions(bundle, prediction))
entities = bundle.entity_table()
entities.group_by("cohort", "event_observed").len()"""
            ),
            markdown(
                """## TTE versus RUL and survival

Row-level ``time_to_event`` matches RUL when the event is end of useful life;
use `pdmdata.tasks.rul` when you want the ``RUL`` column name and NASA-oriented
helpers. ``entity_table()`` keeps test engines right-censored at the last
observed cycle (no official offset), which is the starting point for survival
analysis. See the survival-analysis notebook for Kaplan-Meier on that table."""
            ),
        ]
    ),
    "survival-analysis.ipynb": notebook(
        [
            markdown(
                """# Survival analysis with C-MAPSS

Survival analysis is a time-to-event method that represents both event times
and right-censored observations. This example combines failed training engines
with test engines censored at their final observed cycle.

## Learning goals

- construct one duration and event indicator per engine;
- understand right censoring;
- compute a Kaplan-Meier estimate without a specialized survival package."""
            ),
            code(SETUP),
            code(
                """pdmdata.download("cmapss")
train = pdmdata.load("cmapss", subset="FD001", split="train")
test = pdmdata.load("cmapss", subset="FD001", split="test")

failed = (
    train.group_by("unit_number")
    .agg(pl.col("cycle").max().alias("duration"))
    .with_columns(
        pl.lit(True).alias("event_observed"),
        pl.lit("train failure").alias("cohort"),
    )
)
censored = (
    test.group_by("unit_number")
    .agg(pl.col("cycle").max().alias("duration"))
    .with_columns(
        pl.lit(False).alias("event_observed"),
        pl.lit("test cutoff").alias("cohort"),
    )
)
survival_data = pl.concat([failed, censored])
survival_data.group_by("cohort").agg(
    pl.len().alias("engines"),
    pl.col("duration").median().alias("median_observed_cycles"),
)"""
            ),
            markdown(
                """## Kaplan-Meier estimator

At each failure time, the conditional survival probability is
`1 - failures / engines_at_risk`. Censored engines reduce later risk sets but
do not create a survival drop."""
            ),
            code(
                """event_table = (
    survival_data.group_by("duration")
    .agg(
        pl.col("event_observed").sum().alias("events"),
        (~pl.col("event_observed")).sum().alias("censored"),
    )
    .sort("duration")
)

durations = event_table["duration"].to_numpy()
events = event_table["events"].to_numpy()
at_risk = np.array([
    survival_data.filter(pl.col("duration") >= duration).height
    for duration in durations
])
conditional = np.where(events > 0, 1 - events / at_risk, 1.0)
estimate = event_table.with_columns(
    pl.Series("at_risk", at_risk),
    pl.Series("survival_probability", np.cumprod(conditional)),
)
estimate.head()"""
            ),
            code(
                """plt.step(
    estimate["duration"].to_numpy(),
    estimate["survival_probability"].to_numpy(),
    where="post",
)
plt.ylim(0, 1.02)
plt.xlabel("Cycles since first observation")
plt.ylabel("Estimated survival probability")
plt.show()"""
            ),
            markdown(
                """## Interpretation and limitations

This mixed train/test cohort is instructional, not an unbiased population:
C-MAPSS test cutoffs were constructed for a benchmark. Covariate-aware survival
models also require careful landmarking to avoid treating future sensor values
as baseline information."""
            ),
        ]
    ),
    "event-sequence-forecasting.ipynb": notebook(
        [
            markdown(
                """# Event forecasting with MAPM

Event forecasting predicts what event comes next. MAPM records timestamped
non-fatal machine errors, so it supports a transparent first-order transition
baseline.

## Learning goals

- create next-event labels within each machine;
- fit a transition table on earlier events;
- compare the model with a global-majority baseline on later events."""
            ),
            code(
                SETUP
                + """
from sklearn.metrics import accuracy_score"""
            ),
            code(
                """pdmdata.download("mapm")
errors = (
    pdmdata.load("mapm", table="errors")
    .sort("machineID", "datetime")
    .with_columns(
        pl.col("errorID").shift(-1).over("machineID").alias("next_error"),
        pl.int_range(pl.len()).over("machineID").alias("event_order"),
        pl.len().over("machineID").alias("event_count"),
    )
    .drop_nulls("next_error")
)
errors.select("machineID", "datetime", "errorID", "next_error").head()"""
            ),
            code(
                """train = errors.filter(pl.col("event_order") < pl.col("event_count") * 0.8)
test = errors.filter(pl.col("event_order") >= pl.col("event_count") * 0.8)

transition = (
    train.group_by("errorID", "next_error")
    .len()
    .sort(["errorID", "len"], descending=[False, True])
    .group_by("errorID", maintain_order=True)
    .first()
    .select("errorID", pl.col("next_error").alias("predicted_error"))
)
majority_error = train["next_error"].mode()[0]
scored = test.join(transition, on="errorID", how="left").with_columns(
    pl.col("predicted_error").fill_null(majority_error)
)
print(
    "Transition accuracy:",
    f"{accuracy_score(scored['next_error'].to_numpy(), scored['predicted_error'].to_numpy()):.3f}",
)
print(
    "Majority accuracy:",
    f"{accuracy_score(scored['next_error'].to_numpy(), np.repeat(majority_error, scored.height)):.3f}",
)"""
            ),
            code(
                """transition.sort("errorID")"""
            ),
            markdown(
                """## Interpretation and limitations

The transition baseline ignores elapsed time, telemetry, maintenance, and
longer history. A useful sequence model should improve on both this baseline
and the global majority while preserving a chronological, machine-aware split."""
            ),
        ]
    ),
}


def main() -> None:
    """Write all task notebooks in stable name order."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    expected = set(NOTEBOOKS)
    for existing in OUTPUT.glob("*.ipynb"):
        if existing.name not in expected:
            existing.unlink()
    for filename, document in sorted(NOTEBOOKS.items()):
        path = OUTPUT / filename
        path.write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
