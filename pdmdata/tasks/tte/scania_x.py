"""Time-to-event task view for SCANIA Component X."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from pdmdata.datasets.scania_x.loader import FEATURES, load
from pdmdata.tasks.common import summary_frame
from pdmdata.tasks.metrics import score_split
from pdmdata.tasks.splits import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
)
from pdmdata.tasks.tte.common import (
    cap_remaining,
    cohort_frame,
    materialize_splits,
    partition_entities,
    require_time_cap,
    restore_test_labels,
    select_features,
)
from pdmdata.tasks.types import TabularSplit

ENTITY_COLUMN = "vehicle_id"
TIME_COLUMN = "time_step"

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
    time_cap: float | None
    train: TabularSplit
    validation: TabularSplit | None
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
    vehicles: Sequence[int] | None = None,
    test_vehicles: Sequence[int] | None = None,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_vehicles: Sequence[int] | None = None,
    time_cap: float | None = None,
    drop_constant_features: bool = False,
    feature_columns: Sequence[str] | None = None,
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
    return score_split(bundle.test_eval_split(), predictions)


__all__ = [
    "CLASSES",
    "COST_MATRIX",
    "ScaniaXTteBundle",
    "WINDOW_BOUNDS",
    "add_tte_targets",
    "challenge_cost",
    "cost_table",
    "eval_scania_x",
    "prepare_scania_x",
]
