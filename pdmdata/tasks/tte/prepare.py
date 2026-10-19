"""Time-to-event task views and shared helpers.

RUL is a TTE profile where the event is end of useful life. This module owns
the C-MAPSS TTE experiment bundle (row-level remaining time plus entity-level
censoring) and re-exports the general TTE column vocabulary and entity-split
helpers from ``pdmdata.tasks.splits``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from pdmdata.datasets.cmapss import (
    load as load_cmapss,
    rul as cmapss_test_end_rul,
)
from pdmdata.datasets.cmapss.loader import SUBSETS
from pdmdata.tasks.cmapss_common import (
    CMAPSS_ENTITY_COLUMN,
    CMAPSS_TIME_COLUMN,
    feature_columns,
    label_failed_rows,
    last_rows_per_entity,
    summary_frame,
)
from pdmdata.tasks.metrics import score_split
from pdmdata.tasks.splits import (
    ENTITY_COLUMN as ENTITY_COLUMN,
    EVENT_OBSERVED_COLUMN as EVENT_OBSERVED_COLUMN,
    TARGET_COLUMN as TARGET_COLUMN,
    TIME_COLUMN as TIME_COLUMN,
    TIME_TO_EVENT_COLUMN as TIME_TO_EVENT_COLUMN,
    filter_entities as filter_entities,
    split_entities as split_entities,
)
from pdmdata.tasks.types import TabularSplit


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
    feature_columns: tuple[str, ...]
    time_cap: int | None
    train: TabularSplit
    validation: TabularSplit | None
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
        return summary_frame(self.train, self.validation, self.test)


def prepare_cmapss(
    subset: str = "FD001",
    *,
    time_cap: int | None = None,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_units: Sequence[int] | None = None,
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

    train_labeled = label_failed_rows(train_raw, time_cap=time_cap)
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
    validation_split: TabularSplit | None
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


def eval_cmapss(
    bundle: CmapssTteBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> dict[str, float]:
    """Score predictions on the official last-row remaining-time protocol."""
    return score_split(bundle.test_eval_split(), predictions)
