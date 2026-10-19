"""Shared RUL and TTE task-view helpers, with C-MAPSS column defaults."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pdmdata.datasets.cmapss.loader import OPERATION_COLUMNS, SENSOR_COLUMNS
from pdmdata.tasks.splits import EVENT_OBSERVED_COLUMN, TIME_TO_EVENT_COLUMN
from pdmdata.tasks.types import TabularSplit

CMAPSS_ENTITY_COLUMN = "unit_number"
CMAPSS_TIME_COLUMN = "cycle"


def drop_constant_columns(
    frame: pl.DataFrame,
    columns: Sequence[str],
) -> tuple[str, ...]:
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
) -> tuple[str, ...]:
    """Select C-MAPSS feature columns from a labeled training frame."""
    columns = list(SENSOR_COLUMNS)
    if include_operating_settings:
        columns = list(OPERATION_COLUMNS) + columns
    missing = [column for column in columns if column not in train.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
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


def summary_frame(
    train: TabularSplit,
    validation: TabularSplit | None,
    test: TabularSplit,
) -> pl.DataFrame:
    """Inventory row counts for the prepared splits."""
    rows = [summary_row("train", train), summary_row("test", test)]
    if validation is not None:
        rows.insert(1, summary_row("validation", validation))
    return pl.DataFrame(rows)


def label_rul_rows(
    frame: pl.DataFrame,
    *,
    rul_cap: int | None,
    rul_column: str = "RUL",
) -> pl.DataFrame:
    """Cap run-to-failure ``RUL`` and mirror it into observed TTE columns."""
    labeled = apply_time_cap(frame, rul_cap, rul_column)
    return labeled.with_columns(
        pl.col(rul_column).alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    )


def label_failed_rows(
    frame: pl.DataFrame,
    *,
    time_cap: int | None,
    rul_column: str = "RUL",
) -> pl.DataFrame:
    """Replace run-to-failure ``RUL`` with observed, capped TTE columns."""
    labeled = frame.with_columns(
        pl.col(rul_column).alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    ).drop(rul_column)
    return apply_time_cap(labeled, time_cap, TIME_TO_EVENT_COLUMN)


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
