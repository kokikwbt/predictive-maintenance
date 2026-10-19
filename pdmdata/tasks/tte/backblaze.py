"""Time-to-event task view for Backblaze drive snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from pdmdata.datasets.backblaze import load
from pdmdata.tasks.tte.common import (
    cap_remaining,
    cohort_frame,
    materialize_splits,
    partition_entities,
    require_time_cap,
    restore_test_labels,
    score_remaining,
    select_features,
    summary_frame,
)
from pdmdata.tasks.tte.prepare import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
)
from pdmdata.tasks.types import TabularSplit

ENTITY_COLUMN = "serial_number"
TIME_COLUMN = "date"
_EXCLUDED = {"date", "serial_number", "model", "failure"}


@dataclass(frozen=True)
class BackblazeTteBundle:
    """Model-ready Backblaze time-to-event splits.

    The event is the first day ``failure`` is 1. Drives with no failure
    flag in the loaded files are right-censored at their last snapshot.
    That censoring is the edge of the loaded window, not a lifetime
    survival label. ``time_to_event`` and ``duration`` are whole days.
    """

    variant: str
    feature_columns: tuple[str, ...]
    time_cap: float | None
    train: TabularSplit
    validation: TabularSplit | None
    test: TabularSplit
    test_time: pl.DataFrame
    entity_records: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Held-out drive-days with days until failure or censoring."""
        return restore_test_labels(self.test, self.test_time)

    def entity_table(self) -> pl.DataFrame:
        """One row per drive.

        ``duration`` is days from the first snapshot to the event or
        censoring day.
        """
        return self.entity_records

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared row-level splits."""
        return summary_frame(self.train, self.validation, self.test)


def prepare_backblaze(
    variant: str | None = None,
    *,
    serial_numbers: Sequence[str] | None = None,
    test_serials: Sequence[str] | None = None,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_serials: Sequence[str] | None = None,
    time_cap: float | None = None,
    drop_constant_features: bool = False,
    feature_columns: Sequence[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BackblazeTteBundle:
    """Build a Backblaze time-to-event bundle for one downloaded quarter.

    The ``failure`` column is the event indicator and is never a feature.
    ``model`` is omitted because it is categorical. Date bounds are
    inclusive and follow ``pdmdata.datasets.backblaze.load``.
    """
    cap = require_time_cap(time_cap)
    frame = load(
        variant,
        lazy=False,
        start_date=start_date,
        end_date=end_date,
    )
    if serial_numbers is not None:
        selected = [str(serial) for serial in serial_numbers]
        frame = frame.filter(pl.col(ENTITY_COLUMN).is_in(selected))
    if frame.is_empty():
        raise ValueError("No Backblaze rows matched the selection")
    labeled, resolved_variant = _label_drives(frame, variant)
    entities = labeled[ENTITY_COLUMN].unique().sort().to_list()
    train_ids, validation_ids, test_ids = partition_entities(
        entities,
        test_entities=test_serials,
        test_fraction=test_fraction,
        validation_entities=validation_serials,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    candidates = list(
        feature_columns or _default_features(labeled)
    )
    train_rows = labeled.filter(pl.col(ENTITY_COLUMN).is_in(list(train_ids)))
    features = select_features(
        train_rows,
        candidates,
        drop_constant_features=drop_constant_features,
    )
    durations = _durations(labeled)
    capped = cap_remaining(labeled, cap)
    train, validation, test, test_time = materialize_splits(
        capped,
        entity_column=ENTITY_COLUMN,
        time_column=TIME_COLUMN,
        features=features,
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
    )
    cohort = cohort_frame(ENTITY_COLUMN, train_ids, validation_ids, test_ids)
    entity_records = durations.join(
        cohort, on=ENTITY_COLUMN, how="left"
    ).sort(ENTITY_COLUMN)
    return BackblazeTteBundle(
        variant=resolved_variant,
        feature_columns=features,
        time_cap=cap,
        train=train,
        validation=validation,
        test=test,
        test_time=test_time,
        entity_records=entity_records,
    )


def eval_backblaze(
    bundle: BackblazeTteBundle,
    predictions: Sequence[float] | Mapping[str, float],
) -> dict:
    """Score days remaining on held-out Backblaze drives."""
    return score_remaining(bundle.test_eval_split(), predictions)


def _resolved_variant(variant: str | None) -> str:
    from pdmdata.catalog import load_metadata

    download = load_metadata("backblaze")["download"]
    if variant is None:
        return download["default_variant"]
    return variant


def _default_features(frame: pl.DataFrame) -> list[str]:
    smart = sorted(
        column
        for column in frame.columns
        if column.startswith("smart_") and frame.schema[column].is_numeric()
    )
    columns = []
    if "capacity_bytes" in frame.columns:
        columns.append("capacity_bytes")
    columns.extend(smart)
    if not columns:
        raise ValueError("No numeric Backblaze feature columns were found")
    return columns


def _label_drives(
    frame: pl.DataFrame, variant: str | None
) -> tuple[pl.DataFrame, str]:
    required = {ENTITY_COLUMN, TIME_COLUMN, "failure"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Backblaze frame is missing: {}".format(", ".join(sorted(missing)))
        )
    failed = (
        frame.filter(pl.col("failure") == 1)
        .group_by(ENTITY_COLUMN)
        .agg(pl.col(TIME_COLUMN).min().alias("failure_date"))
    )
    ends = frame.group_by(ENTITY_COLUMN).agg(
        pl.col(TIME_COLUMN).max().alias("last_date")
    )
    horizons = (
        ends.join(failed, on=ENTITY_COLUMN, how="left")
        .with_columns(
            pl.when(pl.col("failure_date").is_not_null())
            .then(pl.col("failure_date"))
            .otherwise(pl.col("last_date"))
            .alias("horizon"),
            pl.col("failure_date").is_not_null().alias(EVENT_OBSERVED_COLUMN),
        )
        .select(ENTITY_COLUMN, "horizon", EVENT_OBSERVED_COLUMN)
    )
    labeled = (
        frame.join(horizons, on=ENTITY_COLUMN, how="left")
        .filter(pl.col(TIME_COLUMN) <= pl.col("horizon"))
        .with_columns(
            (pl.col("horizon") - pl.col(TIME_COLUMN))
            .dt.total_days()
            .alias(TIME_TO_EVENT_COLUMN)
        )
        .drop("horizon", "failure")
    )
    return labeled, _resolved_variant(variant)


def _durations(labeled: pl.DataFrame) -> pl.DataFrame:
    return (
        labeled.sort(ENTITY_COLUMN, TIME_COLUMN)
        .group_by(ENTITY_COLUMN, maintain_order=True)
        .agg(
            pl.col(TIME_TO_EVENT_COLUMN).first().alias("duration"),
            pl.col(EVENT_OBSERVED_COLUMN).first(),
        )
    )


__all__ = [
    "BackblazeTteBundle",
    "eval_backblaze",
    "prepare_backblaze",
]
