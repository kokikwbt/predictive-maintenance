"""Official IDA 2024 Industrial Challenge misclassification cost."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


CLASSES = (0, 1, 2, 3, 4)
# COST_MATRIX[actual][predicted]; values from 2024_IDA_challenge_v2.pdf.
COST_MATRIX = (
    (0, 7, 8, 9, 10),
    (200, 0, 7, 8, 9),
    (300, 200, 0, 7, 8),
    (400, 300, 200, 0, 7),
    (500, 400, 300, 200, 0),
)


def _classes(values: Sequence[int] | pl.Series, name: str) -> list[int]:
    items = values.to_list() if isinstance(values, pl.Series) else list(values)
    if any(item not in CLASSES or isinstance(item, bool) for item in items):
        raise ValueError("{} must contain classes 0-4 only".format(name))
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
