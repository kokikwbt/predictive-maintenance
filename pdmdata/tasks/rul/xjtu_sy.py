"""Remaining useful life (RUL) task view for XJTU-SY bearings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from pdmdata.datasets.xjtu_sy import load_many
from pdmdata.tasks.common import (
    XJTU_SY_DEFAULT_FEATURES,
    XJTU_SY_ENTITY_COLUMN,
    XJTU_SY_TIME_COLUMN,
    label_rul_rows,
    select_xjtu_sy_bearings,
    split_xjtu_sy_test_bearings,
    summary_frame,
    xjtu_sy_feature_columns,
)
from pdmdata.tasks.metrics import score_split
from pdmdata.tasks.splits import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit

ENTITY_COLUMN = XJTU_SY_ENTITY_COLUMN
TIME_COLUMN = XJTU_SY_TIME_COLUMN
TARGET_COLUMN = "RUL"
DEFAULT_FEATURES = XJTU_SY_DEFAULT_FEATURES
_features = xjtu_sy_feature_columns
_select_bearings = select_xjtu_sy_bearings
_split_test = split_xjtu_sy_test_bearings


@dataclass(frozen=True)
class XjtuSyRulBundle:
    """Model-ready XJTU-SY RUL splits from run-to-failure bearings.

    There is no official censored test fleet. Held-out bearings form ``test``
    (features only); ``test_eval_split()`` restores full-trajectory ``RUL`` for
    every snapshot so scoring is not limited to the trivial final ``RUL=0``.
    """

    condition: str | None
    feature_columns: tuple[str, ...]
    rul_cap: int | None
    train: TabularSplit
    validation: TabularSplit | None
    test: TabularSplit
    test_rul: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """All test snapshots joined with held-out ``RUL`` labels."""
        labeled = self.test.frame.join(
            self.test_rul,
            on=[ENTITY_COLUMN, TIME_COLUMN],
            how="left",
            validate="1:1",
            maintain_order="left",
        ).with_columns(
            pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN),
            pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
        )
        return TabularSplit(
            frame=labeled,
            feature_columns=self.feature_columns,
            target_column=TARGET_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared splits."""
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
    rul_cap: int | None = None,
    drop_constant_features: bool = False,
    feature_columns: Sequence[str] | None = None,
) -> XjtuSyRulBundle:
    """Build an XJTU-SY RUL experiment bundle.

    Parameters
    ----------
    condition:
        Optional operating-condition directory filter
        (``35Hz12kN``, ``37.5Hz11kN``, ``40Hz10kN``).
    bearings:
        Explicit bearing ids to include; defaults to locally available
        bearings (optionally filtered by ``condition``).
    test_bearings / test_fraction:
        Held-out bearings for evaluation. Explicit ids override the fraction.
    validation_bearings / validation_fraction:
        Hold-out within the remaining training bearings.
    rul_cap:
        Optional piecewise ``min(RUL, cap)`` on labeled rows.
    """
    if rul_cap is not None and (
        isinstance(rul_cap, bool) or not isinstance(rul_cap, int) or rul_cap < 1
    ):
        raise ValueError("rul_cap must be a positive integer or None")
    if not 0.0 < test_fraction < 1.0 and test_bearings is None:
        raise ValueError("test_fraction must satisfy 0 < f < 1")

    selected = select_xjtu_sy_bearings(bearings=bearings, condition=condition)
    frame = load_many(selected, with_rul=True)
    labeled = label_rul_rows(frame, rul_cap=rul_cap)

    features = xjtu_sy_feature_columns(
        labeled,
        feature_columns=feature_columns,
        drop_constant_features=drop_constant_features,
    )

    train_pool, test_ids = split_xjtu_sy_test_bearings(
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
    test_rul = test_labeled.select(
        ENTITY_COLUMN, TIME_COLUMN, TARGET_COLUMN
    ).sort(ENTITY_COLUMN, TIME_COLUMN)
    test_frame = test_labeled.select(
        [ENTITY_COLUMN, TIME_COLUMN, *features]
    )
    return XjtuSyRulBundle(
        condition=condition,
        feature_columns=features,
        rul_cap=rul_cap,
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
        test_rul=test_rul,
    )


def eval_xjtu_sy(
    bundle: XjtuSyRulBundle,
    predictions: Sequence[float] | Mapping[str, float],
) -> dict[str, float]:
    """Score predictions on the held-out XJTU-SY trajectories.

    A sequence must align with ``test_eval_split()`` row order. A mapping
    keyed by bearing id reuses one value for every snapshot of that bearing.
    """
    return score_split(bundle.test_eval_split(), predictions)


__all__ = [
    "DEFAULT_FEATURES",
    "ENTITY_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "XjtuSyRulBundle",
    "eval_xjtu_sy",
    "prepare_xjtu_sy",
]
