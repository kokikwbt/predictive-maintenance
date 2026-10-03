"""Remaining useful life (RUL) task view for C-MAPSS.

RUL is a time-to-event profile where the event is end of useful life and the
target at each row is remaining operating cycles. Loaders stay source-faithful;
this module builds an explicit Polars-tabular experiment bundle for modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import polars as pl

from pdmdata.datasets.cmapss import load as load_cmapss
from pdmdata.datasets.cmapss import rul as cmapss_test_end_rul
from pdmdata.datasets.cmapss.loader import SUBSETS
from pdmdata.tasks.cmapss_common import (
    CMAPSS_ENTITY_COLUMN,
    CMAPSS_TIME_COLUMN,
    align_predictions,
    apply_time_cap,
    feature_columns,
    last_rows_per_entity,
    summary_row,
)
from pdmdata.tasks.metrics import nasa_score, regression_metrics
from pdmdata.tasks.tte import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit


ENTITY_COLUMN = CMAPSS_ENTITY_COLUMN
TIME_COLUMN = CMAPSS_TIME_COLUMN
TARGET_COLUMN = "RUL"


@dataclass(frozen=True)
class CmapssRulBundle:
    """Model-ready C-MAPSS RUL splits built from source-faithful loaders.

    Train and validation rows include ``RUL`` (and the TTE alias
    ``time_to_event``) plus ``event_observed``. Test feature rows intentionally
    omit the target; use ``test_end_rul`` or ``test_eval_split()`` for the
    official final-observation protocol.
    """

    subset: str
    feature_columns: Tuple[str, ...]
    rul_cap: Optional[int]
    train: TabularSplit
    validation: Optional[TabularSplit]
    test: TabularSplit
    test_end_rul: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Last test row per unit, joined with official end-of-trajectory RUL.

        This is the standard C-MAPSS scoring view: one prediction opportunity
        per test engine at its final observed cycle.
        """
        last_rows = last_rows_per_entity(self.test.frame)
        labeled = last_rows.join(
            self.test_end_rul,
            on=ENTITY_COLUMN,
            how="left",
            validate="1:1",
            maintain_order="left",
        ).with_columns(
            pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN),
            pl.lit(False).alias(EVENT_OBSERVED_COLUMN),
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


def prepare_cmapss(
    subset: str = "FD001",
    *,
    rul_cap: Optional[int] = None,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_units: Optional[Sequence[int]] = None,
    drop_constant_features: bool = False,
    include_operating_settings: bool = True,
) -> CmapssRulBundle:
    """Build a C-MAPSS RUL experiment bundle.

    Parameters
    ----------
    subset:
        One of ``FD001`` … ``FD004``.
    rul_cap:
        Optional piecewise cap ``min(RUL, cap)`` applied after constructing
        uncapped remaining life. ``None`` keeps uncapped targets.
    validation_fraction:
        Fraction of training engines held out when ``validation_units`` is
        omitted. Use ``0`` to skip a validation split.
    random_state:
        Shuffle seed for the unit hold-out.
    validation_units:
        Explicit validation engine IDs; overrides ``validation_fraction``.
    drop_constant_features:
        Drop feature columns that are constant on the full training split.
    include_operating_settings:
        Include ``operation_1..3`` among candidate features.
    """
    if subset not in SUBSETS:
        raise ValueError("subset must be FD001, FD002, FD003, or FD004")
    if rul_cap is not None and (
        isinstance(rul_cap, bool) or not isinstance(rul_cap, int) or rul_cap < 1
    ):
        raise ValueError("rul_cap must be a positive integer or None")

    train_raw = load_cmapss(subset, "train", with_rul=True)
    test_raw = load_cmapss(subset, "test", with_rul=False)
    test_end = cmapss_test_end_rul(subset)

    train_labeled = _label_train(train_raw, rul_cap=rul_cap)
    features = feature_columns(
        train_labeled,
        include_operating_settings=include_operating_settings,
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
    validation_split: Optional[TabularSplit]
    if validation_unit_ids:
        validation_frame = filter_entities(
            train_labeled, validation_unit_ids, ENTITY_COLUMN
        )
        validation_split = TabularSplit(
            frame=validation_frame,
            feature_columns=features,
            target_column=TARGET_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )
    else:
        validation_split = None

    test_frame = test_raw.select(
        [ENTITY_COLUMN, TIME_COLUMN, *features]
    )
    return CmapssRulBundle(
        subset=subset,
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
        test_end_rul=test_end,
    )


def apply_rul_cap(
    frame: pl.DataFrame,
    rul_cap: Optional[int],
    column: str = TARGET_COLUMN,
) -> pl.DataFrame:
    """Return a copy with an optional piecewise RUL cap."""
    return apply_time_cap(frame, rul_cap, column)


def evaluate_test_predictions(
    bundle: CmapssRulBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> Dict[str, float]:
    """Score predictions on the official last-row test protocol.

    ``predictions`` may be a sequence aligned with ``test_eval_split()`` row
    order (sorted ``unit_number``) or a mapping ``unit_number -> prediction``.
    """
    eval_split = bundle.test_eval_split()
    units = eval_split.groups().to_list()
    actual = eval_split.y().to_numpy()
    pred = align_predictions(units, predictions)
    return regression_metrics(actual, pred)


def _label_train(frame: pl.DataFrame, *,
                 rul_cap: Optional[int]) -> pl.DataFrame:
    labeled = frame.with_columns(
        pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    )
    labeled = apply_rul_cap(labeled, rul_cap, TARGET_COLUMN)
    return labeled.with_columns(
        pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN)
    )


# Re-export metrics for callers that import them from ``pdmdata.tasks.rul``.
__all__ = [
    "CmapssRulBundle",
    "ENTITY_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "apply_rul_cap",
    "evaluate_test_predictions",
    "nasa_score",
    "prepare_cmapss",
    "regression_metrics",
]
