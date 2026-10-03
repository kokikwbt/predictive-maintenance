"""Time-to-event task views and shared helpers.

RUL is a TTE profile where the event is end of useful life. This module owns
the general TTE column vocabulary, entity-split helpers, and the C-MAPSS TTE
experiment bundle (row-level remaining time plus entity-level censoring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from pdmdata.datasets.cmapss import load as load_cmapss
from pdmdata.datasets.cmapss import rul as cmapss_test_end_rul
from pdmdata.datasets.cmapss.loader import SUBSETS
from pdmdata.tasks.cmapss_common import (
    CMAPSS_ENTITY_COLUMN,
    CMAPSS_TIME_COLUMN,
    align_predictions,
    apply_time_cap,
    drop_constant_columns,
    feature_columns,
    last_rows_per_entity,
    summary_row,
)
from pdmdata.tasks.metrics import regression_metrics
from pdmdata.tasks.types import TabularSplit


# Generic TTE vocabulary (dataset adapters may keep native names for
# entity/time and still expose these target / censoring columns).
ENTITY_COLUMN = "entity"
TIME_COLUMN = "time"
TIME_TO_EVENT_COLUMN = "time_to_event"
EVENT_OBSERVED_COLUMN = "event_observed"

TARGET_COLUMN = TIME_TO_EVENT_COLUMN


def split_entities(
    entities: Sequence[int],
    *,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_entities: Optional[Sequence[int]] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Split entity IDs into train and validation groups.

    When ``validation_entities`` is given it is used as the hold-out set and
    must be a subset of ``entities``. Otherwise entities are shuffled with
    ``random_state`` and the last fraction is held out. A zero fraction keeps
    every entity in train and returns an empty validation tuple.
    """
    unique = tuple(sorted({int(value) for value in entities}))
    if not unique:
        raise ValueError("entities must be non-empty")
    if validation_entities is not None:
        holdout = tuple(sorted({int(value) for value in validation_entities}))
        unknown = set(holdout) - set(unique)
        if unknown:
            raise ValueError(
                "validation_entities not present in entities: "
                "{}".format(sorted(unknown))
            )
        if set(holdout) == set(unique):
            raise ValueError(
                "validation_entities must leave at least one training entity"
            )
        train = tuple(value for value in unique if value not in set(holdout))
        return train, holdout
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= f < 1")
    if validation_fraction == 0.0:
        return unique, ()
    order = list(unique)
    rng = np.random.default_rng(random_state)
    rng.shuffle(order)
    holdout_count = max(1, int(round(len(order) * validation_fraction)))
    if holdout_count >= len(order):
        holdout_count = len(order) - 1
    holdout = tuple(sorted(order[-holdout_count:]))
    train = tuple(sorted(order[:-holdout_count]))
    return train, holdout


def filter_entities(
    frame: pl.DataFrame,
    entities: Sequence[int],
    entity_column: str,
) -> pl.DataFrame:
    """Return rows whose entity ID is in ``entities``, preserving order."""
    if not entities:
        return frame.clear()
    return frame.filter(pl.col(entity_column).is_in(list(entities)))


@dataclass(frozen=True)
class CmapssTteBundle:
    """Model-ready C-MAPSS time-to-event splits.

    The event is engine failure / end of useful life. Row-level targets use
    ``time_to_event`` (remaining cycles) with ``event_observed``. Train engines
    failed (observed). NASA test engines are right-censored in the observed
    window; ``test_eval_split()`` attaches official remaining time only for
    scoring, while ``entity_table()`` keeps true censoring durations.
    """

    subset: str
    feature_columns: Tuple[str, ...]
    time_cap: Optional[int]
    train: TabularSplit
    validation: Optional[TabularSplit]
    test: TabularSplit
    test_end_time: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Last NASA test row per unit with official remaining time.

        ``event_observed`` is False: failure was not seen in the test window.
        The joined ``time_to_event`` is evaluation information from the
        official RUL file, not an in-window observation.
        """
        last_rows = last_rows_per_entity(self.test.frame)
        labeled = last_rows.join(
            self.test_end_time,
            on=CMAPSS_ENTITY_COLUMN,
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
            entity_column=CMAPSS_ENTITY_COLUMN,
            time_column=CMAPSS_TIME_COLUMN,
        )

    def entity_table(self) -> pl.DataFrame:
        """One row per engine with duration and censoring indicator.

        Training engines contribute observed failures
        (``duration = max(cycle)``, ``event_observed=True``). NASA test engines
        contribute right-censored rows at the last observed cycle
        (``event_observed=False``). Official remaining-life offsets are not
        added to test durations; use ``test_eval_split()`` for that protocol.
        """
        train_raw = load_cmapss(self.subset, "train", with_rul=False)
        test_raw = load_cmapss(self.subset, "test", with_rul=False)
        failed = (
            train_raw.group_by(CMAPSS_ENTITY_COLUMN)
            .agg(pl.col(CMAPSS_TIME_COLUMN).max().alias("duration"))
            .with_columns(
                pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
                pl.lit("train").alias("cohort"),
            )
            .sort(CMAPSS_ENTITY_COLUMN)
        )
        censored = (
            test_raw.group_by(CMAPSS_ENTITY_COLUMN)
            .agg(pl.col(CMAPSS_TIME_COLUMN).max().alias("duration"))
            .with_columns(
                pl.lit(False).alias(EVENT_OBSERVED_COLUMN),
                pl.lit("test").alias("cohort"),
            )
            .sort(CMAPSS_ENTITY_COLUMN)
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


def prepare_cmapss(
    subset: str = "FD001",
    *,
    time_cap: Optional[int] = None,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_units: Optional[Sequence[int]] = None,
    drop_constant_features: bool = False,
    include_operating_settings: bool = True,
) -> CmapssTteBundle:
    """Build a C-MAPSS time-to-event experiment bundle.

    Parameters
    ----------
    subset:
        One of ``FD001`` … ``FD004``.
    time_cap:
        Optional piecewise cap ``min(time_to_event, cap)``. ``None`` keeps
        uncapped remaining time.
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
    if time_cap is not None and (
        isinstance(time_cap, bool)
        or not isinstance(time_cap, int)
        or time_cap < 1
    ):
        raise ValueError("time_cap must be a positive integer or None")

    train_raw = load_cmapss(subset, "train", with_rul=True)
    test_raw = load_cmapss(subset, "test", with_rul=False)
    test_end = cmapss_test_end_rul(subset).rename(
        {"RUL": TIME_TO_EVENT_COLUMN}
    )

    train_labeled = _label_failed_rows(train_raw, time_cap=time_cap)
    features = feature_columns(
        train_labeled,
        include_operating_settings=include_operating_settings,
        drop_constant_features=drop_constant_features,
    )

    units = train_labeled[CMAPSS_ENTITY_COLUMN].unique().sort().to_list()
    train_units, validation_unit_ids = split_entities(
        units,
        validation_fraction=validation_fraction,
        random_state=random_state,
        validation_entities=validation_units,
    )
    train_frame = filter_entities(
        train_labeled, train_units, CMAPSS_ENTITY_COLUMN
    )
    validation_split: Optional[TabularSplit]
    if validation_unit_ids:
        validation_frame = filter_entities(
            train_labeled, validation_unit_ids, CMAPSS_ENTITY_COLUMN
        )
        validation_split = TabularSplit(
            frame=validation_frame,
            feature_columns=features,
            target_column=TARGET_COLUMN,
            entity_column=CMAPSS_ENTITY_COLUMN,
            time_column=CMAPSS_TIME_COLUMN,
        )
    else:
        validation_split = None

    test_frame = test_raw.select(
        [CMAPSS_ENTITY_COLUMN, CMAPSS_TIME_COLUMN, *features]
    )
    return CmapssTteBundle(
        subset=subset,
        feature_columns=features,
        time_cap=time_cap,
        train=TabularSplit(
            frame=train_frame,
            feature_columns=features,
            target_column=TARGET_COLUMN,
            entity_column=CMAPSS_ENTITY_COLUMN,
            time_column=CMAPSS_TIME_COLUMN,
        ),
        validation=validation_split,
        test=TabularSplit(
            frame=test_frame,
            feature_columns=features,
            target_column=None,
            entity_column=CMAPSS_ENTITY_COLUMN,
            time_column=CMAPSS_TIME_COLUMN,
        ),
        test_end_time=test_end,
    )


def evaluate_test_predictions(
    bundle: CmapssTteBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> Dict[str, float]:
    """Score predictions on the official last-row remaining-time protocol."""
    eval_split = bundle.test_eval_split()
    units = eval_split.groups().to_list()
    actual = eval_split.y().to_numpy()
    pred = align_predictions(units, predictions)
    return regression_metrics(actual, pred)


def _label_failed_rows(
    frame: pl.DataFrame,
    *,
    time_cap: Optional[int],
) -> pl.DataFrame:
    """Attach TTE columns for run-to-failure training engines."""
    labeled = frame.with_columns(
        pl.col("RUL").alias(TIME_TO_EVENT_COLUMN),
        pl.lit(True).alias(EVENT_OBSERVED_COLUMN),
    ).drop("RUL")
    labeled = apply_time_cap(labeled, time_cap, TIME_TO_EVENT_COLUMN)
    return labeled
