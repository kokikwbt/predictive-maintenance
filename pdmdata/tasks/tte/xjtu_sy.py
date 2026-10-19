"""Time-to-event (TTE) task view for XJTU-SY bearings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from pdmdata.datasets.xjtu_sy import load_many
from pdmdata.tasks.cmapss_common import label_failed_rows, summary_frame
from pdmdata.tasks.metrics import score_split
from pdmdata.tasks.rul.xjtu_sy import (
    ENTITY_COLUMN,
    TIME_COLUMN,
    _features,
    _select_bearings,
    _split_test,
)
from pdmdata.tasks.tte.common import cohort_frame
from pdmdata.tasks.tte.prepare import (
    EVENT_OBSERVED_COLUMN,
    TARGET_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit


@dataclass(frozen=True)
class XjtuSyTteBundle:
    """Model-ready XJTU-SY time-to-event splits.

    All bearings are run-to-failure (``event_observed=True``). Held-out
    bearings form feature-only ``test`` rows; ``test_eval_split()`` restores
    ``time_to_event`` for full-trajectory scoring.
    """

    condition: str | None
    feature_columns: tuple[str, ...]
    time_cap: int | None
    train: TabularSplit
    validation: TabularSplit | None
    test: TabularSplit
    test_time: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """All test snapshots with held-out ``time_to_event`` labels."""
        labeled = self.test.frame.join(
            self.test_time,
            on=[ENTITY_COLUMN, TIME_COLUMN],
            how="left",
            validate="1:1",
            maintain_order="left",
        ).with_columns(
            pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
        )
        return TabularSplit(
            frame=labeled,
            feature_columns=self.feature_columns,
            target_column=TARGET_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )

    def entity_table(self) -> pl.DataFrame:
        """One row per bearing with observed failure duration.

        Splits keep every snapshot of their bearings, so the last ``cycle``
        is the full run length and no source files are reread.
        """
        train_ids = self.train.groups().unique().to_list()
        validation_ids = (
            self.validation.groups().unique().to_list()
            if self.validation is not None
            else []
        )
        test_ids = self.test.groups().unique().to_list()
        splits = [
            split
            for split in (self.train, self.validation, self.test)
            if split is not None
        ]
        frame = pl.concat(
            [split.frame.select(ENTITY_COLUMN, TIME_COLUMN) for split in splits]
        )
        table = (
            frame.group_by(ENTITY_COLUMN)
            .agg(pl.col(TIME_COLUMN).max().alias("duration"))
            .with_columns(pl.lit(True).alias(EVENT_OBSERVED_COLUMN))
            .sort(ENTITY_COLUMN)
        )
        cohort = cohort_frame(
            ENTITY_COLUMN, train_ids, validation_ids, test_ids
        )
        return table.join(cohort, on=ENTITY_COLUMN, how="left")

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared row-level splits."""
        return summary_frame(self.train, self.validation, self.test)


def prepare_xjtu_sy(
    *,
    condition: str | None = None,
    bearings: Sequence[str] | None = None,
    test_bearings: Sequence[str] | None = None,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_bearings: Sequence[str] | None = None,
    time_cap: int | None = None,
    drop_constant_features: bool = False,
    feature_columns: Sequence[str] | None = None,
) -> XjtuSyTteBundle:
    """Build an XJTU-SY time-to-event experiment bundle."""
    if time_cap is not None and (
        isinstance(time_cap, bool)
        or not isinstance(time_cap, int)
        or time_cap < 1
    ):
        raise ValueError("time_cap must be a positive integer or None")
    if not 0.0 < test_fraction < 1.0 and test_bearings is None:
        raise ValueError("test_fraction must satisfy 0 < f < 1")

    selected = _select_bearings(bearings=bearings, condition=condition)
    frame = load_many(selected, with_rul=True)
    labeled = label_failed_rows(frame, time_cap=time_cap)
    features = _features(
        labeled,
        feature_columns=feature_columns,
        drop_constant_features=drop_constant_features,
    )

    train_pool, test_ids = _split_test(
        selected,
        test_bearings=test_bearings,
        test_fraction=test_fraction,
        random_state=random_state,
    )
    train_ids, validation_ids = split_entities(
        train_pool,
        validation_fraction=validation_fraction,
        random_state=random_state,
        validation_entities=validation_bearings,
    )

    train_frame = filter_entities(labeled, train_ids, ENTITY_COLUMN)
    validation_split: TabularSplit | None
    if validation_ids:
        validation_split = TabularSplit(
            frame=filter_entities(labeled, validation_ids, ENTITY_COLUMN),
            feature_columns=features,
            target_column=TARGET_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )
    else:
        validation_split = None

    test_labeled = filter_entities(labeled, test_ids, ENTITY_COLUMN)
    test_time = test_labeled.select(
        ENTITY_COLUMN, TIME_COLUMN, TIME_TO_EVENT_COLUMN
    ).sort(ENTITY_COLUMN, TIME_COLUMN)
    test_frame = test_labeled.select(
        [ENTITY_COLUMN, TIME_COLUMN, *features]
    )
    return XjtuSyTteBundle(
        condition=condition,
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
        test_time=test_time,
    )


def eval_xjtu_sy(
    bundle: XjtuSyTteBundle,
    predictions: Sequence[float] | Mapping[str, float],
) -> dict[str, float]:
    """Score predictions on held-out XJTU-SY trajectories."""
    return score_split(bundle.test_eval_split(), predictions)



__all__ = [
    "XjtuSyTteBundle",
    "eval_xjtu_sy",
    "prepare_xjtu_sy",
]
