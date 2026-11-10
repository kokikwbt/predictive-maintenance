"""SCANIA Component X TTE labels and IDA challenge cost helpers."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

# Upper bounds of the provider's time windows before repair, for classes 4..1.
# The provider does not state boundary inclusion; windows are half-open here.
WINDOW_BOUNDS = ((4, 6.0), (3, 12.0), (2, 24.0), (1, 48.0))

CLASSES = (0, 1, 2, 3, 4)
# COST_MATRIX[actual][predicted]; values from 2024_IDA_challenge_v2.pdf.
COST_MATRIX = (
    (0, 7, 8, 9, 10),
    (200, 0, 7, 8, 9),
    (300, 200, 0, 7, 8),
    (400, 300, 200, 0, 7),
    (500, 400, 300, 200, 0),
)


def add_tte_targets(
    readouts: pl.DataFrame | pl.LazyFrame, tte: pl.DataFrame
) -> pl.DataFrame | pl.LazyFrame:
    """Join train repair records and derive per-readout targets.

    Adds ``time_to_event`` (study end minus ``time_step``), ``event_observed``
    (``in_study_repair``), and ``class_label`` using the provider's windows.
    Censored readouts closer than 48 time steps to the end of observation
    have an unknown class and receive a null label.
    """
    lazy = isinstance(readouts, pl.LazyFrame)
    records = tte.lazy().select(
        "vehicle_id",
        "length_of_study_time_step",
        pl.col("in_study_repair").alias("event_observed"),
    )
    remaining = pl.col("length_of_study_time_step") - pl.col("time_step")
    label = pl.when(remaining >= WINDOW_BOUNDS[-1][1]).then(pl.lit(0))
    for value, upper in WINDOW_BOUNDS:
        label = label.when(
            (pl.col("event_observed") == 1) & (remaining < upper)
        ).then(pl.lit(value))
    frame = (
        readouts.lazy()
        .join(records, on="vehicle_id", how="left", validate="m:1")
        .with_columns(
            remaining.alias("time_to_event"),
            label.otherwise(None).cast(pl.UInt8).alias("class_label"),
        )
        .drop("length_of_study_time_step")
    )
    return frame if lazy else frame.collect()


def _classes(values: Sequence[int] | pl.Series, name: str) -> list[int]:
    items = values.to_list() if isinstance(values, pl.Series) else list(values)
    if any(item not in CLASSES or isinstance(item, bool) for item in items):
        raise ValueError(f"{name} must contain classes 0-4 only")
    return items


def challenge_cost(
    y_true: Sequence[int] | pl.Series,
    y_pred: Sequence[int] | pl.Series,
) -> int:
    """Return the total challenge cost (lower is better).

    Late or missed alarms (predicted < actual) cost far more than early or
    unnecessary ones (predicted > actual).
    """
    actual = _classes(y_true, "y_true")
    predicted = _classes(y_pred, "y_pred")
    if len(actual) != len(predicted):
        raise ValueError("y_true and y_pred must have the same length")
    return sum(COST_MATRIX[a][p] for a, p in zip(actual, predicted))


def cost_table() -> pl.DataFrame:
    """Return the cost matrix as a long table of actual/predicted classes."""
    return pl.DataFrame(
        [
            {"actual": a, "predicted": p, "cost": COST_MATRIX[a][p]}
            for a in CLASSES
            for p in CLASSES
        ],
        schema={"actual": pl.UInt8, "predicted": pl.UInt8, "cost": pl.Int64},
    )


__all__ = [
    "CLASSES",
    "COST_MATRIX",
    "WINDOW_BOUNDS",
    "add_tte_targets",
    "challenge_cost",
    "cost_table",
]
