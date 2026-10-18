"""Time-to-event task view for SCANIA Component X."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import polars as pl

from pdmdata.datasets.scania_x import FEATURES, load
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


ENTITY_COLUMN = "vehicle_id"
TIME_COLUMN = "time_step"


@dataclass(frozen=True)
class ScaniaXTteBundle:
    """Model-ready SCANIA Component X time-to-event splits.

    Targets use the training repair table. ``time_to_event`` is
    ``length_of_study_time_step - time_step``. ``event_observed`` is true
    when Component X was repaired and false when the vehicle is
    right-censored. Official validation and test files only provide
    time-window classes, so this bundle holds out training vehicles
    instead of treating those classes as remaining time.
    """

    feature_columns: tuple[str, ...]
    time_cap: Optional[float]
    train: TabularSplit
    validation: Optional[TabularSplit]
    test: TabularSplit
    test_time: pl.DataFrame
    entity_records: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Held-out training readouts with remaining time restored.

        Censored vehicles keep time until the end of observation. That
        value is not a latent repair time.
        """
        return restore_test_labels(self.test, self.test_time)

    def entity_table(self) -> pl.DataFrame:
        """One row per training vehicle.

        ``duration`` is ``length_of_study_time_step``, measured from the
        start of Component X rather than from the first readout.
        """
        return self.entity_records

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared row-level splits."""
        return summary_frame(self.train, self.validation, self.test)


def prepare_scania_x(
    *,
    vehicles: Optional[Sequence[int]] = None,
    test_vehicles: Optional[Sequence[int]] = None,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_vehicles: Optional[Sequence[int]] = None,
    time_cap: Optional[float] = None,
    drop_constant_features: bool = False,
    feature_columns: Optional[Sequence[str]] = None,
) -> ScaniaXTteBundle:
    """Build a SCANIA Component X time-to-event bundle from train readouts.

    ``time_cap`` caps remaining time in the same anonymized time-step
    units as ``time_step``. Specification columns stay out of the feature
    matrix because they are categorical.
    """
    cap = require_time_cap(time_cap)
    labeled = load("train", with_labels=True, lazy=False)
    if vehicles is not None:
        selected = {int(vehicle) for vehicle in vehicles}
        labeled = labeled.filter(pl.col(ENTITY_COLUMN).is_in(list(selected)))
    if labeled.is_empty():
        raise ValueError("No SCANIA training readouts matched the selection")
    labeled = labeled.with_columns(
        pl.col(EVENT_OBSERVED_COLUMN).cast(pl.Boolean)
    )
    if labeled[TIME_TO_EVENT_COLUMN].null_count():
        raise ValueError("Training readouts are missing repair records")
    if labeled[TIME_TO_EVENT_COLUMN].min() < 0:
        raise ValueError("time_step exceeds length_of_study_time_step")

    records = load("train", "tte").join(
        labeled.select(ENTITY_COLUMN).unique(),
        on=ENTITY_COLUMN,
        how="inner",
    )
    entities = records[ENTITY_COLUMN].unique().sort().to_list()
    train_ids, validation_ids, test_ids = partition_entities(
        entities,
        test_entities=test_vehicles,
        test_fraction=test_fraction,
        validation_entities=validation_vehicles,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    candidates = list(feature_columns or FEATURES)
    train_rows = labeled.filter(pl.col(ENTITY_COLUMN).is_in(list(train_ids)))
    features = select_features(
        train_rows,
        candidates,
        drop_constant_features=drop_constant_features,
    )
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
    cohort = cohort_frame(
        ENTITY_COLUMN, train_ids, validation_ids, test_ids
    )
    entity_records = (
        records.select(
            ENTITY_COLUMN,
            pl.col("length_of_study_time_step").alias("duration"),
            (pl.col("in_study_repair") == 1).alias(EVENT_OBSERVED_COLUMN),
        )
        .join(cohort, on=ENTITY_COLUMN, how="left")
        .sort(ENTITY_COLUMN)
    )
    return ScaniaXTteBundle(
        feature_columns=features,
        time_cap=cap,
        train=train,
        validation=validation,
        test=test,
        test_time=test_time,
        entity_records=entity_records,
    )


def eval_scania_x(
    bundle: ScaniaXTteBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> dict:
    """Score remaining time on held-out SCANIA training vehicles."""
    return score_remaining(bundle.test_eval_split(), predictions)


__all__ = [
    "ScaniaXTteBundle",
    "eval_scania_x",
    "prepare_scania_x",
]
