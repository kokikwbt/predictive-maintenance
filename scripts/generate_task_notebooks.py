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

ROOT = Path.cwd().resolve()
if not (ROOT / "datasets").is_dir():
    ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import datasets"""


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
                "display_name": "Python (pmdata)",
                "language": "python",
                "name": "pmdata",
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
                """datasets.download("gfd")

def vibration_windows(condition, load, window_size=256):
    frame = datasets.load("gfd", condition=condition, load=load)
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
                """datasets.download("gfd")

def window_features(condition, load, window_size=256):
    frame = datasets.load("gfd", condition=condition, load=load)
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
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score"""
            ),
            code(
                """downloaded = datasets.download("oyicd")
files = sorted(Path(downloaded["extracted"]).rglob("*_mode*.csv"))
print(f"Recordings: {len(files)}")

def summarize_recording(path):
    frame = pl.read_csv(path)
    numeric = [
        column for column, dtype in frame.schema.items()
        if dtype.is_numeric()
    ]
    mode = int(re.search(r"mode(\\d+)", path.name).group(1))
    return frame.select(
        *[pl.col(column).mean().alias(f"{column}_mean") for column in numeric],
        *[pl.col(column).std().alias(f"{column}_std") for column in numeric],
    ).with_columns(
        pl.lit(path.name).alias("recording"),
        pl.lit(mode).alias("mode"),
    )

recordings = pl.concat([summarize_recording(path) for path in files], how="diagonal_relaxed")
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

Mode is an operating condition, not a fault label. Because recording time and
degradation may be correlated, a chronological split is preferable to random
rows. A production model should also test whether mode recognition transfers
between machines and component ages."""
            ),
        ]
    ),
    "condition-estimation.ipynb": notebook(
        [
            markdown(
                """# Condition estimation with naval propulsion data

Condition estimation predicts a continuous health quantity. CBM provides
compressor and turbine degradation coefficients as explicit regression targets.

## Learning goals

- distinguish condition estimation from discrete fault classification;
- train a multivariate regression baseline;
- inspect absolute error across the degradation range."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split"""
            ),
            code(
                """datasets.download("cbm")
data = datasets.load("cbm")
target = "kMc"
features = [column for column in data.columns if column not in {"kMc", "kMt"}]
data.select(features + [target]).describe()"""
            ),
            code(
                """indices = np.arange(data.height)
train_index, test_index = train_test_split(indices, test_size=0.2, random_state=0)
model = ExtraTreesRegressor(n_estimators=200, random_state=0, n_jobs=-1)
model.fit(data[train_index].select(features).to_numpy(), data[train_index][target].to_numpy())
prediction = model.predict(data[test_index].select(features).to_numpy())
actual = data[test_index][target].to_numpy()
print(f"Compressor-degradation MAE: {mean_absolute_error(actual, prediction):.5f}")"""
            ),
            code(
                """plt.scatter(actual, prediction, s=8, alpha=0.4)
limits = [min(actual.min(), prediction.min()), max(actual.max(), prediction.max())]
plt.plot(limits, limits, "--", color="black")
plt.xlabel("Actual compressor degradation coefficient")
plt.ylabel("Predicted coefficient")
plt.show()"""
            ),
            markdown(
                """## Interpretation and limitations

CBM is simulated steady-state coverage rather than a run-to-failure sequence.
It supports condition regression, but not event timing or survival analysis.
Random splitting measures interpolation and should not be described as
cross-vessel or temporal generalization."""
            ),
        ]
    ),
    "remaining-useful-life-prediction.ipynb": notebook(
        [
            markdown(
                """# Remaining useful life prediction with C-MAPSS

RUL prediction estimates the remaining cycles at each observation. C-MAPSS
training engines run to failure, while the test set ends before failure and
provides the true RUL at each engine's final observation.

## Learning goals

- construct row-level RUL without crossing engine boundaries;
- split training data by engine;
- evaluate final-observation predictions with the official test targets."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error"""
            ),
            code(
                """datasets.download("cmapss")
train = datasets.load("cmapss", subset="FD001", split="train")
test = datasets.load("cmapss", subset="FD001", split="test")
test_rul = datasets.load("cmapss", subset="FD001", split="rul")

train = train.with_columns(
    (pl.col("cycle").max().over("unit_number") - pl.col("cycle")).alias("RUL")
)
train.select("unit_number", "cycle", "RUL").head()"""
            ),
            code(
                """units = train["unit_number"].unique().sort().to_numpy()
development_units = units[: int(len(units) * 0.8)]
validation_units = units[int(len(units) * 0.8) :]
features = [
    column for column in train.columns
    if column.startswith(("operation_", "sensor_"))
]

development = train.filter(pl.col("unit_number").is_in(development_units))
validation = train.filter(pl.col("unit_number").is_in(validation_units))
model = HistGradientBoostingRegressor(loss="absolute_error", random_state=0)
model.fit(development.select(features).to_numpy(), development["RUL"].to_numpy())
validation_prediction = model.predict(validation.select(features).to_numpy())
print(
    "Validation MAE:",
    f"{mean_absolute_error(validation['RUL'].to_numpy(), validation_prediction):.1f} cycles",
)"""
            ),
            code(
                """model.fit(train.select(features).to_numpy(), train["RUL"].to_numpy())
last_test_rows = test.sort("cycle").group_by("unit_number", maintain_order=True).last()
prediction = np.maximum(model.predict(last_test_rows.select(features).to_numpy()), 0)
actual = test_rul["RUL"].to_numpy()
print(f"Official test MAE: {mean_absolute_error(actual, prediction):.1f} cycles")
print(f"Official test RMSE: {root_mean_squared_error(actual, prediction):.1f} cycles")"""
            ),
            markdown(
                """## Interpretation and limitations

The uncapped linear RUL target assumes degradation starts at the first cycle.
Common C-MAPSS studies use a piecewise cap and specialized asymmetric scores;
those choices must be reported because they materially change the task."""
            ),
        ]
    ),
    "time-to-event-prediction.ipynb": notebook(
        [
            markdown(
                """# Time-to-event prediction with C-MAPSS

Time-to-event prediction is the broad problem of estimating when a defined
event will occur. Here the event is engine failure and elapsed time is measured
in operating cycles.

## Learning goals

- define the event and time origin explicitly;
- construct cycles-to-failure with Polars;
- evaluate a regression baseline with an engine-level split."""
            ),
            code(
                SETUP
                + """
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error"""
            ),
            code(
                """datasets.download("cmapss")
data = datasets.load("cmapss", subset="FD001", split="train")
data = data.with_columns(
    (pl.col("cycle").max().over("unit_number") - pl.col("cycle"))
    .alias("time_to_event_cycles"),
    pl.lit(True).alias("event_observed"),
)
data.select("unit_number", "cycle", "time_to_event_cycles", "event_observed").head()"""
            ),
            code(
                """plt.hist(data["time_to_event_cycles"].to_numpy(), bins=60)
plt.xlabel("Operating cycles until failure")
plt.ylabel("Observations")
plt.show()"""
            ),
            code(
                """unit_numbers = data["unit_number"].unique().sort().to_numpy()
split_index = int(len(unit_numbers) * 0.8)
training_units = unit_numbers[:split_index]
test_units = unit_numbers[split_index:]
train = data.filter(pl.col("unit_number").is_in(training_units))
test = data.filter(pl.col("unit_number").is_in(test_units))
features = [
    column for column in data.columns
    if column.startswith(("operation_", "sensor_"))
]

model = HistGradientBoostingRegressor(loss="absolute_error", random_state=0)
model.fit(train.select(features).to_numpy(), train["time_to_event_cycles"].to_numpy())
prediction = np.maximum(model.predict(test.select(features).to_numpy()), 0)
print(
    "Engine-held-out MAE:",
    f"{mean_absolute_error(test['time_to_event_cycles'].to_numpy(), prediction):.1f} cycles",
)"""
            ),
            markdown(
                """## Time-to-event versus survival analysis

This regression uses only observed failures and predicts one point estimate.
Survival analysis instead estimates an event-time distribution and can include
engines whose failure is not observed before data collection ends. See the
survival-analysis notebook for that formulation."""
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
                """datasets.download("cmapss")
train = datasets.load("cmapss", subset="FD001", split="train")
test = datasets.load("cmapss", subset="FD001", split="test")

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
                """datasets.download("mapm")
errors = (
    datasets.load("mapm", table="errors")
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
