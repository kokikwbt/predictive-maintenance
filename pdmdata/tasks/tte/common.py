"""Shared split and scoring helpers for censored TTE bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import polars as pl

from pdmdata.tasks.cmapss_common import (
    align_predictions,
    drop_constant_columns,
    summary_row,
)
from pdmdata.tasks.metrics import regression_metrics
from pdmdata.tasks.tte.prepare import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit


def require_time_cap(time_cap: float | int | None) -> float | None:
    """Return a positive cap, or ``None`` when remaining time stays uncapped."""
    if time_cap is None:
        return None
    if isinstance(time_cap, bool) or not isinstance(time_cap, (int, float)):
        raise ValueError("time_cap must be a positive number or None")
    if time_cap <= 0:
        raise ValueError("time_cap must be a positive number or None")
    return float(time_cap)


def cap_remaining(
    frame: pl.DataFrame,
    time_cap: float | None,
    column: str = TIME_TO_EVENT_COLUMN,
) -> pl.DataFrame:
    """Apply ``min(column, time_cap)`` without casting to an integer type."""
    if time_cap is None:
        return frame
    return frame.with_columns(
        pl.min_horizontal(pl.col(column), pl.lit(time_cap)).alias(column)
    )


def select_features(
    frame: pl.DataFrame,
    candidates: Sequence[str],
    *,
    drop_constant_features: bool,
) -> tuple[str, ...]:
    """Keep numeric candidate columns, optionally dropping train constants."""
    missing = [column for column in candidates if column not in frame.columns]
    if missing:
        raise ValueError(
            "Missing feature columns: {}".format(", ".join(missing))
        )
    non_numeric = [
        column
        for column in candidates
        if not frame.schema[column].is_numeric()
    ]
    if non_numeric:
        raise ValueError(
            "Feature columns must be numeric: {}".format(
                ", ".join(non_numeric)
            )
        )
    columns = list(candidates)
    if drop_constant_features:
        columns = list(drop_constant_columns(frame, columns))
        if not columns:
            raise ValueError("All candidate features are constant on train")
    return tuple(columns)


def partition_entities(
    entities: Sequence[int | str],
    *,
    test_entities: Sequence[int | str] | None,
    test_fraction: float,
    validation_entities: Sequence[int | str] | None,
    validation_fraction: float,
    random_state: int,
) -> tuple[tuple[int | str, ...], tuple[int | str, ...], tuple[int | str, ...]]:
    """Split entities into train, validation, and test groups.

    Explicit test or validation ids override the corresponding fraction.
    A one-entity training pool yields an empty validation group.
    """
    if test_entities is not None:
        train_pool, test_ids = split_entities(
            entities, validation_entities=test_entities
        )
    else:
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must satisfy 0 < f < 1")
        train_pool, test_ids = split_entities(
            entities,
            validation_fraction=test_fraction,
            random_state=random_state,
        )
    if len(train_pool) < 2 and validation_entities is None:
        validation_fraction = 0.0
    train_ids, validation_ids = split_entities(
        train_pool,
        validation_fraction=validation_fraction,
        random_state=random_state,
        validation_entities=validation_entities,
    )
    return train_ids, validation_ids, test_ids


def cohort_frame(
    entity_column: str,
    train_ids: Sequence[int | str],
    validation_ids: Sequence[int | str],
    test_ids: Sequence[int | str],
) -> pl.DataFrame:
    """Map each entity id to ``train``, ``validation``, or ``test``."""
    rows = (
        [(entity, "train") for entity in train_ids]
        + [(entity, "validation") for entity in validation_ids]
        + [(entity, "test") for entity in test_ids]
    )
    return pl.DataFrame(
        rows,
        schema=[entity_column, "cohort"],
        orient="row",
    )


def materialize_splits(
    labeled: pl.DataFrame,
    *,
    entity_column: str,
    time_column: str,
    features: Sequence[str],
    train_ids: Sequence[int | str],
    validation_ids: Sequence[int | str],
    test_ids: Sequence[int | str],
) -> tuple[TabularSplit, TabularSplit | None, TabularSplit, pl.DataFrame]:
    """Build labeled train/validation splits and feature-only test rows."""
    feature_columns = tuple(features)
    keep = [
        entity_column,
        time_column,
        *feature_columns,
        TIME_TO_EVENT_COLUMN,
        EVENT_OBSERVED_COLUMN,
    ]

    def take(ids: Sequence[int | str]) -> pl.DataFrame:
        return (
            filter_entities(labeled, ids, entity_column)
            .select(keep)
            .sort(entity_column, time_column)
        )

    train_frame = take(train_ids)
    validation_split: TabularSplit | None
    if validation_ids:
        validation_split = TabularSplit(
            frame=take(validation_ids),
            feature_columns=feature_columns,
            target_column=TIME_TO_EVENT_COLUMN,
            entity_column=entity_column,
            time_column=time_column,
        )
    else:
        validation_split = None
    test_labeled = take(test_ids)
    test_time = test_labeled.select(
        entity_column,
        time_column,
        TIME_TO_EVENT_COLUMN,
        EVENT_OBSERVED_COLUMN,
    )
    test_frame = test_labeled.select(
        [entity_column, time_column, *feature_columns]
    )
    train = TabularSplit(
        frame=train_frame,
        feature_columns=feature_columns,
        target_column=TIME_TO_EVENT_COLUMN,
        entity_column=entity_column,
        time_column=time_column,
    )
    test = TabularSplit(
        frame=test_frame,
        feature_columns=feature_columns,
        target_column=None,
        entity_column=entity_column,
        time_column=time_column,
    )
    return train, validation_split, test, test_time


def restore_test_labels(
    test: TabularSplit,
    test_time: pl.DataFrame,
) -> TabularSplit:
    """Join held-out remaining time onto feature-only test rows."""
    labeled = test.frame.join(
        test_time,
        on=[test.entity_column, test.time_column],
        how="left",
        validate="1:1",
        maintain_order="left",
    )
    return TabularSplit(
        frame=labeled,
        feature_columns=test.feature_columns,
        target_column=TIME_TO_EVENT_COLUMN,
        entity_column=test.entity_column,
        time_column=test.time_column,
    )


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


def score_remaining(
    split: TabularSplit,
    predictions: Sequence[float] | Mapping[int | str, float],
) -> dict:
    """Score row-aligned remaining-time predictions.

    A mapping keyed by entity id reuses one value for every row of that
    entity. A sequence must follow ``split`` row order.
    """
    entities = split.groups().to_list()
    actual = split.y().to_numpy()
    predicted = align_predictions(entities, predictions)
    return regression_metrics(actual, predicted)
