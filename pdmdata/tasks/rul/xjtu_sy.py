"""Remaining useful life (RUL) task view for XJTU-SY bearings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import polars as pl

from pdmdata.datasets.xjtu_sy import load_many
from pdmdata.datasets.xjtu_sy.loader import BEARING_INFO, available_bearings
from pdmdata.tasks.cmapss_common import (
    apply_time_cap,
    drop_constant_columns,
    summary_row,
)
from pdmdata.tasks.metrics import regression_metrics
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit


ENTITY_COLUMN = "bearing"
TIME_COLUMN = "cycle"
TARGET_COLUMN = "RUL"
DEFAULT_FEATURES = (
    "rms_horizontal",
    "rms_vertical",
    "rpm",
    "load_kn",
)


@dataclass(frozen=True)
class XjtuSyRulBundle:
    """Model-ready XJTU-SY RUL splits from run-to-failure bearings.

    There is no official censored test fleet. Held-out bearings form ``test``
    (features only); ``test_eval_split()`` restores full-trajectory ``RUL`` for
    every snapshot so scoring is not limited to the trivial final ``RUL=0``.
    """

    condition: Optional[str]
    feature_columns: Tuple[str, ...]
    rul_cap: Optional[int]
    train: TabularSplit
    validation: Optional[TabularSplit]
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
    rul_cap: Optional[int] = None,
    drop_constant_features: bool = False,
    feature_columns: Optional[Sequence[str]] = None,
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

    selected = _select_bearings(bearings=bearings, condition=condition)
    frame = load_many(selected, with_rul=True)
    labeled = _label(frame, rul_cap=rul_cap)

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
) -> Dict[str, float]:
    """Score predictions on the held-out XJTU-SY trajectories.

    A sequence must align with ``test_eval_split()`` row order. A mapping
    keyed by bearing id reuses one value for every snapshot of that bearing.
    """
    import numpy as np

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


def _select_bearings(
    *,
    bearings: Optional[Sequence[str]],
    condition: Optional[str],
) -> Tuple[str, ...]:
    available = available_bearings()
    if bearings is not None:
        selected = tuple(bearings)
        unknown = [name for name in selected if name not in BEARING_INFO]
        if unknown:
            raise ValueError("Unknown bearings: {}".format(unknown))
        missing = [name for name in selected if name not in available]
        if missing:
            raise FileNotFoundError(
                "Bearings not available locally: {}".format(missing)
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
                "Bearings outside condition {!r}: {}".format(condition, bad)
            )
    return selected


def _split_test(
    bearings: Sequence[str],
    *,
    test_bearings: Optional[Sequence[str]],
    test_fraction: float,
    random_state: int,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if test_bearings is not None:
        test_ids = tuple(sorted({str(name) for name in test_bearings}))
        unknown = set(test_ids) - set(bearings)
        if unknown:
            raise ValueError(
                "test_bearings not in selected set: {}".format(sorted(unknown))
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
    # split_entities names the hold-out "validation"; here it is the test set.
    return train_pool, test_ids


def _label(frame: pl.DataFrame, *, rul_cap: Optional[int]) -> pl.DataFrame:
    labeled = frame.with_columns(
        pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    )
    labeled = apply_time_cap(labeled, rul_cap, TARGET_COLUMN)
    return labeled.with_columns(
        pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN)
    )


def _features(
    train: pl.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]],
    drop_constant_features: bool,
) -> Tuple[str, ...]:
    columns = (
        tuple(feature_columns) if feature_columns is not None
        else DEFAULT_FEATURES
    )
    missing = [name for name in columns if name not in train.columns]
    if missing:
        raise ValueError("Missing feature columns: {}".format(missing))
    if drop_constant_features:
        columns = drop_constant_columns(train, columns)
        if not columns:
            raise ValueError("All candidate features are constant on train")
    return tuple(columns)


__all__ = [
    "DEFAULT_FEATURES",
    "ENTITY_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "XjtuSyRulBundle",
    "eval_xjtu_sy",
    "prepare_xjtu_sy",
]
