"""Shared RUL and TTE task-view helpers across datasets."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from pdmdata.datasets.cmapss.loader import OPERATION_COLUMNS, SENSOR_COLUMNS
from pdmdata.datasets.xjtu_sy.loader import BEARING_INFO, available_bearings
from pdmdata.tasks.splits import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit

CMAPSS_ENTITY_COLUMN = "unit_number"
CMAPSS_TIME_COLUMN = "cycle"

NCMAPSS_ENTITY_COLUMN = "unit_number"
NCMAPSS_TIME_COLUMN = "cycle"
_NCMAPSS_AUX_COLUMNS = ("Fc", "hs")
_NCMAPSS_FLIGHT_COLUMNS = ("alt", "Mach", "TRA", "T2")

XJTU_SY_ENTITY_COLUMN = "bearing"
XJTU_SY_TIME_COLUMN = "cycle"
XJTU_SY_DEFAULT_FEATURES = (
    "rms_horizontal",
    "rms_vertical",
    "rpm",
    "load_kn",
)


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


def cmapss_feature_columns(
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


feature_columns = cmapss_feature_columns


def ncmapss_feature_columns(
    train: pl.DataFrame,
    *,
    include_flight_conditions: bool,
    include_aux: bool,
    drop_constant_features: bool,
) -> tuple[str, ...]:
    """Select numeric N-CMAPSS feature columns from a training frame."""
    exclude = {
        NCMAPSS_ENTITY_COLUMN,
        NCMAPSS_TIME_COLUMN,
        "RUL",
        TIME_TO_EVENT_COLUMN,
        EVENT_OBSERVED_COLUMN,
    }
    if not include_aux:
        exclude.update(_NCMAPSS_AUX_COLUMNS)
    if not include_flight_conditions:
        exclude.update(_NCMAPSS_FLIGHT_COLUMNS)
    columns = [
        name
        for name in train.columns
        if name not in exclude and train.schema[name].is_numeric()
    ]
    if not columns:
        raise ValueError("No numeric feature columns available")
    if drop_constant_features:
        columns = list(drop_constant_columns(train, columns))
        if not columns:
            raise ValueError("All candidate features are constant on train")
    return tuple(columns)


def xjtu_sy_feature_columns(
    train: pl.DataFrame,
    *,
    feature_columns: Sequence[str] | None,
    drop_constant_features: bool,
) -> tuple[str, ...]:
    """Select XJTU-SY feature columns from a training frame."""
    columns = (
        tuple(feature_columns)
        if feature_columns is not None
        else XJTU_SY_DEFAULT_FEATURES
    )
    missing = [name for name in columns if name not in train.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if drop_constant_features:
        columns = drop_constant_columns(train, columns)
        if not columns:
            raise ValueError("All candidate features are constant on train")
    return tuple(columns)


def select_xjtu_sy_bearings(
    *,
    bearings: Sequence[str] | None,
    condition: str | None,
) -> tuple[str, ...]:
    """Resolve and validate locally available XJTU-SY bearings."""
    available = available_bearings()
    if bearings is not None:
        selected = tuple(bearings)
        unknown = [name for name in selected if name not in BEARING_INFO]
        if unknown:
            raise ValueError(f"Unknown bearings: {unknown}")
        missing = [name for name in selected if name not in available]
        if missing:
            raise FileNotFoundError(
                f"Bearings not available locally: {missing}"
            )
    else:
        selected = tuple(
            name
            for name in available
            if condition is None
            or BEARING_INFO[name]["condition"] == condition
        )
    if len(selected) < 2:
        raise ValueError(
            "Need at least two local bearings to form train and test"
        )
    if condition is not None:
        bad = [
            name
            for name in selected
            if BEARING_INFO[name]["condition"] != condition
        ]
        if bad:
            raise ValueError(
                f"Bearings outside condition {condition!r}: {bad}"
            )
    return selected


def split_xjtu_sy_test_bearings(
    bearings: Sequence[str],
    *,
    test_bearings: Sequence[str] | None,
    test_fraction: float,
    random_state: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split XJTU-SY bearings into a training pool and held-out test set."""
    if test_bearings is not None:
        test_ids = tuple(sorted({str(name) for name in test_bearings}))
        unknown = set(test_ids) - set(bearings)
        if unknown:
            raise ValueError(
                f"test_bearings not in selected set: {sorted(unknown)}"
            )
        if set(test_ids) == set(bearings):
            raise ValueError(
                "test_bearings must leave at least one training bearing"
            )
        train_pool = tuple(
            name for name in bearings if name not in set(test_ids)
        )
        return train_pool, test_ids
    train_pool, test_ids = split_entities(
        bearings,
        validation_fraction=test_fraction,
        random_state=random_state,
    )
    return train_pool, test_ids


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


def failure_and_censoring_durations(
    failed: Sequence[TabularSplit | None],
    censored: TabularSplit,
    *,
    failed_cohort: str,
) -> pl.DataFrame:
    """One row per entity with its last time index as ``duration``.

    ``failed`` splits hold complete run-to-failure entities (observed
    events); ``censored`` holds right-censored test entities. Splits must
    keep every row of their entities.
    """

    def durations(
        splits: list[TabularSplit], observed: bool, cohort: str
    ) -> pl.DataFrame:
        entity, time = splits[0].entity_column, splits[0].time_column
        return (
            pl.concat([split.frame.select(entity, time) for split in splits])
            .group_by(entity)
            .agg(pl.col(time).max().alias("duration"))
            .with_columns(
                pl.lit(observed).alias(EVENT_OBSERVED_COLUMN),
                pl.lit(cohort).alias("cohort"),
            )
            .sort(entity)
        )

    return pl.concat(
        [
            durations(
                [split for split in failed if split is not None],
                True,
                failed_cohort,
            ),
            durations([censored], False, "test"),
        ],
        how="vertical_relaxed",
    )


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


__all__ = [
    "CMAPSS_ENTITY_COLUMN",
    "CMAPSS_TIME_COLUMN",
    "NCMAPSS_ENTITY_COLUMN",
    "NCMAPSS_TIME_COLUMN",
    "XJTU_SY_DEFAULT_FEATURES",
    "XJTU_SY_ENTITY_COLUMN",
    "XJTU_SY_TIME_COLUMN",
    "apply_time_cap",
    "cmapss_feature_columns",
    "drop_constant_columns",
    "failure_and_censoring_durations",
    "feature_columns",
    "label_failed_rows",
    "label_rul_rows",
    "last_rows_per_entity",
    "ncmapss_feature_columns",
    "select_xjtu_sy_bearings",
    "split_xjtu_sy_test_bearings",
    "summary_frame",
    "summary_row",
    "xjtu_sy_feature_columns",
]
