"""Time-to-event task view for the Azure predictive-maintenance sample."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from pdmdata.datasets.mapm import load
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

ENTITY_COLUMN = "machineID"
TIME_COLUMN = "datetime"
_SENSOR_COLUMNS = ("volt", "rotate", "pressure", "vibration")


@dataclass(frozen=True)
class MapmTteBundle:
    """Model-ready MAPM time-to-event splits.

    The event is the next recorded component failure at or after a
    telemetry timestamp. A machine with no later failure is right-censored
    at its last telemetry time. Maintenance does not reset the clock.
    ``time_to_event`` is hours. ``entity_table`` has one row per
    inter-failure spell, not one row per machine.
    """

    feature_columns: tuple[str, ...]
    time_cap: float | None
    train: TabularSplit
    validation: TabularSplit | None
    test: TabularSplit
    test_time: pl.DataFrame
    entity_records: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Held-out rows with hours until the next failure or censoring."""
        return restore_test_labels(self.test, self.test_time)

    def entity_table(self) -> pl.DataFrame:
        """One row per machine spell.

        ``spell`` starts at 0. ``duration`` is hours from the first
        telemetry row of the spell until its failure or censoring time.
        """
        return self.entity_records

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared row-level splits."""
        return summary_frame(self.train, self.validation, self.test)


def prepare_mapm(
    *,
    machines: Sequence[int] | None = None,
    test_machines: Sequence[int] | None = None,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_machines: Sequence[int] | None = None,
    time_cap: float | None = None,
    drop_constant_features: bool = False,
    feature_columns: Sequence[str] | None = None,
    include_age: bool = True,
) -> MapmTteBundle:
    """Build a MAPM time-to-event bundle from telemetry and failure tables.

    Default features are the four telemetry channels plus numeric ``age``.
    ``model`` is categorical and is not included.
    """
    cap = require_time_cap(time_cap)
    labeled = _label_telemetry(machines=machines, include_age=include_age)
    entities = labeled[ENTITY_COLUMN].unique().sort().to_list()
    train_ids, validation_ids, test_ids = partition_entities(
        entities,
        test_entities=test_machines,
        test_fraction=test_fraction,
        validation_entities=validation_machines,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    candidates = list(
        feature_columns or _default_features(labeled, include_age)
    )
    train_rows = labeled.filter(pl.col(ENTITY_COLUMN).is_in(list(train_ids)))
    features = select_features(
        train_rows,
        candidates,
        drop_constant_features=drop_constant_features,
    )
    durations = _spells(labeled)
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
    ).sort(ENTITY_COLUMN, "spell")
    return MapmTteBundle(
        feature_columns=features,
        time_cap=cap,
        train=train,
        validation=validation,
        test=test,
        test_time=test_time,
        entity_records=entity_records,
    )


def eval_mapm(
    bundle: MapmTteBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> dict:
    """Score hours remaining on held-out MAPM machines."""
    return score_remaining(bundle.test_eval_split(), predictions)


def _default_features(frame: pl.DataFrame, include_age: bool) -> list[str]:
    columns = [column for column in _SENSOR_COLUMNS if column in frame.columns]
    if include_age and "age" in frame.columns:
        columns.append("age")
    if not columns:
        raise ValueError("No MAPM telemetry feature columns were found")
    return columns


def _label_telemetry(
    *,
    machines: Sequence[int] | None,
    include_age: bool,
) -> pl.DataFrame:
    telemetry = load("telemetry")
    failures = load("failures")
    for name, frame in (("telemetry", telemetry), ("failures", failures)):
        missing = {ENTITY_COLUMN, TIME_COLUMN} - set(frame.columns)
        if missing:
            raise ValueError(
                "MAPM {} is missing: {}".format(
                    name, ", ".join(sorted(missing))
                )
            )
    if machines is not None:
        selected = [int(machine) for machine in machines]
        telemetry = telemetry.filter(pl.col(ENTITY_COLUMN).is_in(selected))
        failures = failures.filter(pl.col(ENTITY_COLUMN).is_in(selected))
    if telemetry.is_empty():
        raise ValueError("No MAPM telemetry rows matched the selection")
    duplicates = (
        telemetry.group_by(ENTITY_COLUMN, TIME_COLUMN)
        .len()
        .filter(pl.col("len") > 1)
    )
    if duplicates.height:
        raise ValueError("MAPM telemetry has duplicate machineID/datetime rows")
    failure_times = (
        failures.select(
            ENTITY_COLUMN,
            pl.col(TIME_COLUMN).alias("failure_time"),
        )
        .unique(subset=[ENTITY_COLUMN, "failure_time"])
        .sort(ENTITY_COLUMN, "failure_time")
    )
    labeled = telemetry.sort(ENTITY_COLUMN, TIME_COLUMN).join_asof(
        failure_times,
        left_on=TIME_COLUMN,
        right_on="failure_time",
        by=ENTITY_COLUMN,
        strategy="forward",
        check_sortedness=False,
    )
    censor = telemetry.group_by(ENTITY_COLUMN).agg(
        pl.col(TIME_COLUMN).max().alias("censor_time")
    )
    labeled = (
        labeled.join(censor, on=ENTITY_COLUMN, how="left")
        .with_columns(
            pl.when(pl.col("failure_time").is_not_null())
            .then(pl.col("failure_time"))
            .otherwise(pl.col("censor_time"))
            .alias("horizon"),
            pl.col("failure_time").is_not_null().alias(EVENT_OBSERVED_COLUMN),
        )
        .with_columns(
            (
                (pl.col("horizon") - pl.col(TIME_COLUMN)).dt.total_seconds()
                / 3600.0
            ).alias(TIME_TO_EVENT_COLUMN)
        )
        .drop("failure_time", "censor_time", "horizon")
    )
    if include_age:
        machines_table = load("machines").select(ENTITY_COLUMN, "age")
        labeled = labeled.join(
            machines_table, on=ENTITY_COLUMN, how="left", validate="m:1"
        )
    if labeled[TIME_TO_EVENT_COLUMN].min() < 0:
        raise ValueError("MAPM remaining time is negative")
    return labeled


def _spells(labeled: pl.DataFrame) -> pl.DataFrame:
    ordered = labeled.sort(ENTITY_COLUMN, TIME_COLUMN).with_columns(
        (
            (pl.col(TIME_TO_EVENT_COLUMN) == 0)
            & pl.col(EVENT_OBSERVED_COLUMN)
        )
        .shift(1)
        .over(ENTITY_COLUMN)
        .fill_null(False)
        .cum_sum()
        .over(ENTITY_COLUMN)
        .alias("spell")
    )
    return (
        ordered.group_by(ENTITY_COLUMN, "spell", maintain_order=True)
        .agg(
            pl.col(TIME_TO_EVENT_COLUMN).first().alias("duration"),
            pl.col(EVENT_OBSERVED_COLUMN).first(),
        )
    )


__all__ = [
    "MapmTteBundle",
    "eval_mapm",
    "prepare_mapm",
]
