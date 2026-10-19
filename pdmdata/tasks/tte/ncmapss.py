"""Time-to-event (TTE) task view for N-CMAPSS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from pdmdata.datasets.ncmapss import load as load_ncmapss
from pdmdata.datasets.ncmapss.loader import VARIANTS
from pdmdata.tasks.cmapss_common import (
    align_predictions,
    apply_time_cap,
    last_rows_per_entity,
    summary_row,
)
from pdmdata.tasks.metrics import regression_metrics
from pdmdata.tasks.rul.ncmapss import (
    ENTITY_COLUMN,
    TIME_COLUMN,
    _feature_columns,
)
from pdmdata.tasks.tte.prepare import (
    EVENT_OBSERVED_COLUMN,
    TARGET_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit


@dataclass(frozen=True)
class NcmapssTteBundle:
    """Model-ready N-CMAPSS time-to-event splits.

    Development units are observed failures. Test units are treated as
    right-censored in the observed window; ``test_eval_split()`` attaches
    last-cycle remaining time from ``Y_test`` for scoring only.
    """

    variant: str
    feature_columns: tuple[str, ...]
    time_cap: int | None
    train: TabularSplit
    validation: TabularSplit | None
    test: TabularSplit
    test_end_time: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Last test row per unit with official remaining time."""
        last_rows = last_rows_per_entity(
            self.test.frame,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )
        labeled = last_rows.join(
            self.test_end_time,
            on=ENTITY_COLUMN,
            how="left",
            validate="1:1",
            maintain_order="left",
        ).with_columns(
            pl.lit(False).alias(EVENT_OBSERVED_COLUMN),
        )
        return TabularSplit(
            frame=labeled,
            feature_columns=self.feature_columns,
            target_column=TARGET_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )

    def entity_table(self) -> pl.DataFrame:
        """One row per unit with duration and censoring indicator."""
        train_raw = load_ncmapss(self.variant, "dev", with_rul=False)
        test_raw = load_ncmapss(self.variant, "test", with_rul=False)
        failed = (
            train_raw.group_by(ENTITY_COLUMN)
            .agg(pl.col(TIME_COLUMN).max().alias("duration"))
            .with_columns(
                pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
                pl.lit("dev").alias("cohort"),
            )
            .sort(ENTITY_COLUMN)
        )
        censored = (
            test_raw.group_by(ENTITY_COLUMN)
            .agg(pl.col(TIME_COLUMN).max().alias("duration"))
            .with_columns(
                pl.lit(False).alias(EVENT_OBSERVED_COLUMN),
                pl.lit("test").alias("cohort"),
            )
            .sort(ENTITY_COLUMN)
        )
        return pl.concat([failed, censored], how="vertical_relaxed")

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared row-level splits."""
        rows = [
            summary_row("train", self.train),
            summary_row("test", self.test),
        ]
        if self.validation is not None:
            rows.insert(1, summary_row("validation", self.validation))
        return pl.DataFrame(rows)


def prepare_ncmapss(
    variant: str = "ds01",
    *,
    time_cap: int | None = None,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_units: Sequence[int] | None = None,
    drop_constant_features: bool = False,
    include_flight_conditions: bool = True,
    include_aux: bool = False,
    include_virtual: bool = False,
    include_health_modifiers: bool = False,
) -> NcmapssTteBundle:
    """Build an N-CMAPSS time-to-event experiment bundle."""
    if variant not in VARIANTS:
        raise ValueError(
            "variant must be one of: {}".format(", ".join(VARIANTS))
        )
    if time_cap is not None and (
        isinstance(time_cap, bool)
        or not isinstance(time_cap, int)
        or time_cap < 1
    ):
        raise ValueError("time_cap must be a positive integer or None")

    train_raw = load_ncmapss(
        variant,
        "dev",
        with_rul=True,
        include_virtual=include_virtual,
        include_health_modifiers=include_health_modifiers,
    )
    test_raw = load_ncmapss(
        variant,
        "test",
        with_rul=True,
        include_virtual=include_virtual,
        include_health_modifiers=include_health_modifiers,
    )
    test_end = (
        last_rows_per_entity(
            test_raw,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )
        .select(ENTITY_COLUMN, pl.col("RUL").alias(TIME_TO_EVENT_COLUMN))
        .sort(ENTITY_COLUMN)
    )

    train_labeled = _label_failed_rows(train_raw, time_cap=time_cap)
    features = _feature_columns(
        train_labeled,
        include_flight_conditions=include_flight_conditions,
        include_aux=include_aux,
        drop_constant_features=drop_constant_features,
    )

    units = train_labeled[ENTITY_COLUMN].unique().sort().to_list()
    train_units, validation_unit_ids = split_entities(
        units,
        validation_fraction=validation_fraction,
        random_state=random_state,
        validation_entities=validation_units,
    )
    train_frame = filter_entities(
        train_labeled, train_units, ENTITY_COLUMN
    )
    validation_split: TabularSplit | None
    if validation_unit_ids:
        validation_split = TabularSplit(
            frame=filter_entities(
                train_labeled, validation_unit_ids, ENTITY_COLUMN
            ),
            feature_columns=features,
            target_column=TARGET_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )
    else:
        validation_split = None

    test_frame = test_raw.select([ENTITY_COLUMN, TIME_COLUMN, *features])
    return NcmapssTteBundle(
        variant=variant,
        feature_columns=features,
        time_cap=time_cap,
        train=TabularSplit(
            frame=train_frame,
            feature_columns=features,
            target_column=TARGET_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        ),
        validation=validation_split,
        test=TabularSplit(
            frame=test_frame,
            feature_columns=features,
            target_column=None,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        ),
        test_end_time=test_end,
    )


def eval_ncmapss(
    bundle: NcmapssTteBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> dict[str, float]:
    """Score predictions on the last-row remaining-time protocol."""
    eval_split = bundle.test_eval_split()
    units = eval_split.groups().to_list()
    actual = eval_split.y().to_numpy()
    pred = align_predictions(units, predictions)
    return regression_metrics(actual, pred)


def _label_failed_rows(
    frame: pl.DataFrame,
    *,
    time_cap: int | None,
) -> pl.DataFrame:
    labeled = frame.with_columns(
        pl.col("RUL").alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    ).drop("RUL")
    return apply_time_cap(labeled, time_cap, TIME_TO_EVENT_COLUMN)


__all__ = [
    "NcmapssTteBundle",
    "eval_ncmapss",
    "prepare_ncmapss",
]
