"""Shared C-MAPSS helpers for RUL and TTE task views."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import polars as pl

from pdmdata.datasets.cmapss.loader import OPERATION_COLUMNS, SENSOR_COLUMNS
from pdmdata.tasks.types import TabularSplit


CMAPSS_ENTITY_COLUMN = "unit_number"
CMAPSS_TIME_COLUMN = "cycle"


def drop_constant_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
) -> Tuple[str, ...]:
    """Keep columns that vary on ``frame``; preserve input order."""
    kept = []
    for column in columns:
        series = frame.get_column(column)
        if series.n_unique() > 1:
            kept.append(column)
    return tuple(kept)


def feature_columns(
    train: pl.DataFrame,
    *,
    include_operating_settings: bool,
    drop_constant_features: bool,
) -> Tuple[str, ...]:
    """Select C-MAPSS feature columns from a labeled training frame."""
    columns = list(SENSOR_COLUMNS)
    if include_operating_settings:
        columns = list(OPERATION_COLUMNS) + columns
    missing = [column for column in columns if column not in train.columns]
    if missing:
        raise ValueError("Missing feature columns: {}".format(missing))
    if drop_constant_features:
        columns = list(drop_constant_columns(train, columns))
        if not columns:
            raise ValueError("All candidate features are constant on train")
    return tuple(columns)


def apply_time_cap(
    frame: pl.DataFrame,
    time_cap: int | None,
    column: str,
) -> pl.DataFrame:
    """Return a copy with an optional piecewise remaining-time cap."""
    if time_cap is None:
        return frame
    return frame.with_columns(
        pl.min_horizontal(pl.col(column), pl.lit(time_cap))
        .cast(pl.UInt32)
        .alias(column)
    )


def summary_row(name: str, split: TabularSplit) -> dict:
    """Build one inventory row for a prepared split."""
    return {
        "split": name,
        "rows": split.frame.height,
        "entities": split.groups().n_unique(),
        "features": len(split.feature_columns),
        "has_target": split.target_column is not None,
    }


def last_rows_per_entity(
    frame: pl.DataFrame,
    *,
    entity_column: str = CMAPSS_ENTITY_COLUMN,
    time_column: str = CMAPSS_TIME_COLUMN,
) -> pl.DataFrame:
    """Return the last observation for each entity, sorted by entity ID."""
    return (
        frame.sort(time_column)
        .group_by(entity_column, maintain_order=True)
        .last()
        .sort(entity_column)
    )


def align_predictions(
    units: Sequence[int],
    predictions: Sequence[float] | dict,
) -> np.ndarray:
    """Align sequence or unit-keyed predictions to ``units`` order."""
    if isinstance(predictions, dict):
        try:
            return np.asarray(
                [predictions[unit] for unit in units], dtype=float
            )
        except KeyError as error:
            raise KeyError(
                "Missing prediction for test unit {}".format(error.args[0])
            ) from error
    pred = np.asarray(predictions, dtype=float)
    if pred.shape != (len(units),):
        raise ValueError(
            "Expected {} predictions for test units, got {}".format(
                len(units), pred.size
            )
        )
    return pred
