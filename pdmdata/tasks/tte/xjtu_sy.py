"""Time-to-event (TTE) task view for XJTU-SY bearings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from pdmdata.datasets.xjtu_sy import load_many
from pdmdata.tasks.cmapss_common import apply_time_cap, summary_row
from pdmdata.tasks.metrics import regression_metrics
from pdmdata.tasks.rul.xjtu_sy import (
    ENTITY_COLUMN,
    TIME_COLUMN,
    _features,
    _select_bearings,
    _split_test,
)
from pdmdata.tasks.tte import (
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

    condition: Optional[str]
    feature_columns: Tuple[str, ...]
    time_cap: Optional[int]
    train: TabularSplit
    validation: Optional[TabularSplit]
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
        """One row per bearing with observed failure duration."""
        train_ids = self.train.groups().unique().to_list()
        validation_ids = (
            self.validation.groups().unique().to_list()
            if self.validation is not None
            else []
        )
        test_ids = self.test.groups().unique().to_list()
        frame = load_many(
            [*train_ids, *validation_ids, *test_ids], with_rul=False
        )
        table = (
            frame.group_by(ENTITY_COLUMN)
            .agg(pl.col(TIME_COLUMN).max().alias("duration"))
            .with_columns(pl.lit(True).alias(EVENT_OBSERVED_COLUMN))
            .sort(ENTITY_COLUMN)
        )
        cohort_rows = (
            [(name, "train") for name in train_ids]
            + [(name, "validation") for name in validation_ids]
            + [(name, "test") for name in test_ids]
        )
        cohort = pl.DataFrame(
            cohort_rows,
            schema=[ENTITY_COLUMN, "cohort"],
            orient="row",
        )
        return table.join(cohort, on=ENTITY_COLUMN, how="left")

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared row-level splits."""
        rows = [
            summary_row("train", self.train),
            summary_row("test", self.test),
        ]
        if self.validation is not None:
            rows.insert(1, summary_row("validation", self.validation))
        return pl.DataFrame(rows)


def prepare_xjtu_sy(
    *,
    condition: str | None = None,
    bearings: Optional[Sequence[str]] = None,
    test_bearings: Optional[Sequence[str]] = None,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_bearings: Optional[Sequence[str]] = None,
    time_cap: Optional[int] = None,
    drop_constant_features: bool = False,
    feature_columns: Optional[Sequence[str]] = None,
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
    labeled = _label(frame, time_cap=time_cap)
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
    validation_split: Optional[TabularSplit]
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
) -> Dict[str, float]:
    """Score predictions on held-out XJTU-SY trajectories."""
    eval_split = bundle.test_eval_split()
    entities = eval_split.groups().to_list()
    actual = eval_split.y().to_numpy()
    if isinstance(predictions, Mapping):
        try:
            pred = np.asarray(
                [predictions[entity] for entity in entities], dtype=float
            )
        except KeyError as error:
            raise KeyError(
                "Missing prediction for test bearing {}".format(error.args[0])
            ) from error
    else:
        pred = np.asarray(predictions, dtype=float)
        if pred.shape != (len(entities),):
            raise ValueError(
                "Expected {} predictions for test rows, got {}".format(
                    len(entities), pred.size
                )
            )
    return regression_metrics(actual, pred)


def _label(frame: pl.DataFrame, *, time_cap: Optional[int]) -> pl.DataFrame:
    labeled = frame.with_columns(
        pl.col("RUL").alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    ).drop("RUL")
    return apply_time_cap(labeled, time_cap, TIME_TO_EVENT_COLUMN)


__all__ = [
    "XjtuSyTteBundle",
    "eval_xjtu_sy",
    "prepare_xjtu_sy",
]
