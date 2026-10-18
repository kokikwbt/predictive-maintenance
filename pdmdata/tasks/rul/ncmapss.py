"""Remaining useful life (RUL) task view for N-CMAPSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import polars as pl

from pdmdata.datasets.ncmapss import load as load_ncmapss
from pdmdata.datasets.ncmapss.loader import VARIANTS
from pdmdata.tasks.cmapss_common import (
    align_predictions,
    apply_time_cap,
    drop_constant_columns,
    last_rows_per_entity,
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


ENTITY_COLUMN = "unit_number"
TIME_COLUMN = "cycle"
TARGET_COLUMN = "RUL"
_AUX_COLUMNS = ("Fc", "hs")
_FLIGHT_COLUMNS = ("alt", "Mach", "TRA", "T2")


@dataclass(frozen=True)
class NcmapssRulBundle:
    """Model-ready N-CMAPSS RUL splits.

    Development units form train/validation. The HDF5 ``test`` split provides
    feature rows; last-cycle ``RUL`` from ``Y_test`` is kept for scoring via
    ``test_eval_split()``.
    """

    variant: str
    feature_columns: Tuple[str, ...]
    rul_cap: Optional[int]
    train: TabularSplit
    validation: Optional[TabularSplit]
    test: TabularSplit
    test_end_rul: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Last test row per unit joined with official end-of-window RUL."""
        last_rows = last_rows_per_entity(
            self.test.frame,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )
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


def prepare_ncmapss(
    variant: str = "ds01",
    *,
    rul_cap: Optional[int] = None,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_units: Optional[Sequence[int]] = None,
    drop_constant_features: bool = False,
    include_flight_conditions: bool = True,
    include_aux: bool = False,
    include_virtual: bool = False,
    include_health_modifiers: bool = False,
) -> NcmapssRulBundle:
    """Build an N-CMAPSS RUL experiment bundle.

    Parameters
    ----------
    variant:
        One of the N-CMAPSS subset ids (``ds01``, ``ds02``, …).
    rul_cap:
        Optional piecewise ``min(RUL, cap)``. ``None`` keeps uncapped targets.
    validation_fraction:
        Fraction of development units held out when ``validation_units`` is
        omitted. Use ``0`` to skip validation.
    include_flight_conditions:
        Include ``alt``, ``Mach``, ``TRA``, and ``T2``.
    include_aux:
        Include flight class ``Fc`` and health-state ``hs``.
    include_virtual / include_health_modifiers:
        Forwarded to the N-CMAPSS loader for optional ``X_v`` / ``T`` channels.
    """
    if variant not in VARIANTS:
        raise ValueError(
            "variant must be one of: {}".format(", ".join(VARIANTS))
        )
    if rul_cap is not None and (
        isinstance(rul_cap, bool) or not isinstance(rul_cap, int) or rul_cap < 1
    ):
        raise ValueError("rul_cap must be a positive integer or None")

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
        .select(ENTITY_COLUMN, TARGET_COLUMN)
        .sort(ENTITY_COLUMN)
    )

    train_labeled = _label_train(train_raw, rul_cap=rul_cap)
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

    test_frame = test_raw.select([ENTITY_COLUMN, TIME_COLUMN, *features])
    return NcmapssRulBundle(
        variant=variant,
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


def evaluate_test_predictions(
    bundle: NcmapssRulBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> Dict[str, float]:
    """Score predictions on the last-row N-CMAPSS test protocol."""
    eval_split = bundle.test_eval_split()
    units = eval_split.groups().to_list()
    actual = eval_split.y().to_numpy()
    pred = align_predictions(units, predictions)
    return regression_metrics(actual, pred)


def _label_train(
    frame: pl.DataFrame, *, rul_cap: Optional[int]
) -> pl.DataFrame:
    labeled = frame.with_columns(
        pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    )
    labeled = apply_time_cap(labeled, rul_cap, TARGET_COLUMN)
    return labeled.with_columns(
        pl.col(TARGET_COLUMN).alias(TIME_TO_EVENT_COLUMN)
    )


def _feature_columns(
    train: pl.DataFrame,
    *,
    include_flight_conditions: bool,
    include_aux: bool,
    drop_constant_features: bool,
) -> Tuple[str, ...]:
    exclude = {
        ENTITY_COLUMN,
        TIME_COLUMN,
        TARGET_COLUMN,
        TIME_TO_EVENT_COLUMN,
        EVENT_OBSERVED_COLUMN,
    }
    if not include_aux:
        exclude.update(_AUX_COLUMNS)
    if not include_flight_conditions:
        exclude.update(_FLIGHT_COLUMNS)
    columns = [
        name
        for name in train.columns
        if name not in exclude and train.schema[name].is_numeric()
    ]
    if not columns:
        raise ValueError("No numeric feature columns available")
    if drop_constant_features:
        columns = list(drop_constant_columns(train, columns))
        if not columns:
            raise ValueError("All candidate features are constant on train")
    return tuple(columns)


__all__ = [
    "ENTITY_COLUMN",
    "TARGET_COLUMN",
    "TIME_COLUMN",
    "NcmapssRulBundle",
    "evaluate_test_predictions",
    "prepare_ncmapss",
]
