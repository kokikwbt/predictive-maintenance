"""Time-to-event task view for CARE wind-turbine events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import polars as pl

from pdmdata.datasets.care import load
from pdmdata.tasks.tte.common import (
    cap_remaining,
    require_time_cap,
    restore_test_labels,
    score_remaining,
    select_features,
    summary_frame,
)
from pdmdata.tasks.tte.prepare import (
    EVENT_OBSERVED_COLUMN,
    TIME_TO_EVENT_COLUMN,
    filter_entities,
    split_entities,
)
from pdmdata.tasks.types import TabularSplit


ENTITY_COLUMN = "event_id"
TIME_COLUMN = "time_stamp"
_EXCLUDED = {
    "time_stamp",
    "asset_id",
    "asset",
    "id",
    "train_test",
    "status_type_id",
    "event_id",
    "event_label",
    "event_start",
    "event_end",
}


@dataclass(frozen=True)
class CareTteBundle:
    """Model-ready CARE time-to-event splits for one wind farm.

    An anomaly event is observed at ``event_start``. Rows after that
    onset are omitted. A normal event is right-censored at its last
    timestamp. ``time_to_event`` is hours. The feature-only test set is
    the provider ``prediction`` window, not a random event hold-out.
    """

    wind_farm: str
    feature_columns: tuple[str, ...]
    time_cap: Optional[float]
    train: TabularSplit
    validation: Optional[TabularSplit]
    test: TabularSplit
    test_time: pl.DataFrame
    entity_records: pl.DataFrame

    def test_eval_split(self) -> TabularSplit:
        """Prediction-window rows with hours until onset or censoring."""
        return restore_test_labels(self.test, self.test_time)

    def entity_table(self) -> pl.DataFrame:
        """One row per event.

        For anomalies, ``duration`` is hours from the first timestamp to
        ``event_start``. For normal events it is hours from the first
        timestamp to the last.
        """
        return self.entity_records

    def summary(self) -> pl.DataFrame:
        """Return a compact inventory of the prepared row-level splits."""
        return summary_frame(self.train, self.validation, self.test)


def prepare_care(
    wind_farm: str = "A",
    *,
    event_ids: Optional[Sequence[int]] = None,
    validation_fraction: float = 0.2,
    random_state: int = 0,
    validation_events: Optional[Sequence[int]] = None,
    time_cap: Optional[float] = None,
    drop_constant_features: bool = False,
    feature_columns: Optional[Sequence[str]] = None,
    average_channels_only: bool = True,
) -> CareTteBundle:
    """Build a CARE time-to-event bundle for one wind farm.

    Validation holds out whole events from the ``train`` window. Their
    ``prediction`` rows stay in ``test``. ``status_type_id`` is not a
    feature. With ``average_channels_only``, only ``*_avg`` signals are
    candidates.
    """
    cap = require_time_cap(time_cap)
    if (
        not isinstance(wind_farm, str)
        or wind_farm.upper() not in {"A", "B", "C"}
    ):
        raise ValueError("wind_farm must be A, B, or C")
    farm = wind_farm.upper()
    labeled = _label_events(farm, event_ids=event_ids)
    entities = labeled[ENTITY_COLUMN].unique().sort().to_list()
    if len(entities) < 2 and validation_events is None:
        validation_fraction = 0.0
    train_ids, validation_ids = split_entities(
        entities,
        validation_fraction=validation_fraction,
        random_state=random_state,
        validation_entities=validation_events,
    )
    candidates = list(
        feature_columns or _default_features(labeled, average_channels_only)
    )
    train_rows = labeled.filter(
        (pl.col(ENTITY_COLUMN).is_in(list(train_ids)))
        & (pl.col("train_test") == "train")
    )
    if train_rows.is_empty():
        raise ValueError("CARE training window has no rows before event onset")
    features = select_features(
        train_rows,
        candidates,
        drop_constant_features=drop_constant_features,
    )
    durations = _durations(labeled)
    capped = cap_remaining(labeled, cap)
    train, validation, test, test_time = _windows(
        capped,
        features=features,
        train_ids=train_ids,
        validation_ids=validation_ids,
    )
    cohort = pl.DataFrame(
        (
            [(entity, "train") for entity in train_ids]
            + [(entity, "validation") for entity in validation_ids]
        ),
        schema=[ENTITY_COLUMN, "cohort"],
        orient="row",
    )
    entity_records = durations.join(
        cohort, on=ENTITY_COLUMN, how="left"
    ).sort(ENTITY_COLUMN)
    return CareTteBundle(
        wind_farm=farm,
        feature_columns=features,
        time_cap=cap,
        train=train,
        validation=validation,
        test=test,
        test_time=test_time,
        entity_records=entity_records,
    )


def eval_care(
    bundle: CareTteBundle,
    predictions: Sequence[float] | Mapping[int, float],
) -> dict:
    """Score hours remaining on the CARE prediction window."""
    return score_remaining(bundle.test_eval_split(), predictions)


def _default_features(
    frame: pl.DataFrame, average_channels_only: bool
) -> list[str]:
    numeric = [
        name
        for name, dtype in frame.schema.items()
        if name not in _EXCLUDED and dtype.is_numeric()
    ]
    if average_channels_only:
        numeric = [name for name in numeric if name.endswith("_avg")]
    if not numeric:
        raise ValueError("No numeric CARE signal columns were selected")
    return numeric


def _label_events(
    wind_farm: str,
    *,
    event_ids: Optional[Sequence[int]],
) -> pl.DataFrame:
    events = load(wind_farm=wind_farm, table="events", lazy=False)
    required = {"event_id", "event_label", "event_start"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(
            "CARE event table is missing: {}".format(", ".join(sorted(missing)))
        )
    events = events.with_columns(
        pl.col("event_id").cast(pl.Int64),
        pl.col("event_label")
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_lowercase(),
    )
    if events["event_start"].dtype == pl.String:
        events = events.with_columns(
            pl.col("event_start").str.to_datetime(strict=False)
        )
    labels = set(events["event_label"].unique().to_list())
    unknown = labels - {"anomaly", "normal"}
    if unknown:
        raise ValueError(
            "CARE event_label must be anomaly or normal, found: {}".format(
                ", ".join(sorted(str(label) for label in unknown))
            )
        )
    if event_ids is not None:
        selected = [int(event_id) for event_id in event_ids]
        events = events.filter(pl.col("event_id").is_in(selected))
    if events.is_empty():
        raise ValueError("No CARE events matched the selection")
    if events["event_id"].n_unique() != events.height:
        raise ValueError("CARE event_id values are not unique")
    missing_onset = events.filter(
        (pl.col("event_label") == "anomaly") & pl.col("event_start").is_null()
    )
    if missing_onset.height:
        raise ValueError("Anomaly events are missing event_start")

    frames = []
    info = events.select("event_id", "event_label", "event_start")
    for event_id in info["event_id"].to_list():
        frame = load(
            wind_farm=wind_farm, event_id=int(event_id), lazy=False
        ).with_columns(
            pl.lit(int(event_id)).cast(pl.Int64).alias(ENTITY_COLUMN)
        )
        frames.append(frame)
    series = pl.concat(frames, how="diagonal_relaxed").join(
        info, on=ENTITY_COLUMN, how="left"
    )
    if "train_test" not in series.columns or TIME_COLUMN not in series.columns:
        raise ValueError("CARE series require time_stamp and train_test")
    unexpected = set(series["train_test"].unique().to_list()) - {
        "train",
        "prediction",
    }
    if unexpected:
        raise ValueError(
            "CARE train_test values must be train or prediction, found: "
            "{}".format(", ".join(sorted(str(value) for value in unexpected)))
        )
    ends = series.group_by(ENTITY_COLUMN).agg(
        pl.col(TIME_COLUMN).min().alias("series_start"),
        pl.col(TIME_COLUMN).max().alias("series_end"),
    )
    labeled = series.join(ends, on=ENTITY_COLUMN, how="left").with_columns(
        (pl.col("event_label") == "anomaly").alias(EVENT_OBSERVED_COLUMN),
        pl.when(pl.col("event_label") == "anomaly")
        .then(pl.col("event_start"))
        .otherwise(pl.col("series_end"))
        .alias("horizon"),
    )
    labeled = labeled.filter(
        ~pl.col(EVENT_OBSERVED_COLUMN)
        | (pl.col(TIME_COLUMN) <= pl.col("event_start"))
    )
    if labeled.is_empty():
        raise ValueError("No CARE rows remain at or before event onset")
    anomaly_ids = set(
        info.filter(pl.col("event_label") == "anomaly")["event_id"].to_list()
    )
    kept_ids = set(labeled[ENTITY_COLUMN].unique().to_list())
    dropped = anomaly_ids - kept_ids
    if dropped:
        raise ValueError(
            "Anomaly events have no rows at or before onset: {}".format(
                ", ".join(str(event_id) for event_id in sorted(dropped))
            )
        )
    labeled = labeled.with_columns(
        (
            (pl.col("horizon") - pl.col(TIME_COLUMN)).dt.total_seconds()
            / 3600.0
        ).alias(TIME_TO_EVENT_COLUMN)
    )
    if labeled[TIME_TO_EVENT_COLUMN].min() < 0:
        raise ValueError("CARE remaining time is negative")
    return labeled


def _durations(labeled: pl.DataFrame) -> pl.DataFrame:
    return (
        labeled.sort(ENTITY_COLUMN, TIME_COLUMN)
        .group_by(ENTITY_COLUMN, maintain_order=True)
        .agg(
            (
                (pl.col("horizon").first() - pl.col("series_start").first())
                .dt.total_seconds()
                / 3600.0
            ).alias("duration"),
            pl.col(EVENT_OBSERVED_COLUMN).first(),
        )
    )


def _windows(
    labeled: pl.DataFrame,
    *,
    features: Sequence[str],
    train_ids: Sequence[int],
    validation_ids: Sequence[int],
) -> tuple[TabularSplit, Optional[TabularSplit], TabularSplit, pl.DataFrame]:
    feature_columns = tuple(features)
    keep = [
        ENTITY_COLUMN,
        TIME_COLUMN,
        *feature_columns,
        TIME_TO_EVENT_COLUMN,
        EVENT_OBSERVED_COLUMN,
    ]

    def take(ids: Sequence[int], window: str) -> pl.DataFrame:
        return (
            filter_entities(labeled, ids, ENTITY_COLUMN)
            .filter(pl.col("train_test") == window)
            .select(keep)
            .sort(ENTITY_COLUMN, TIME_COLUMN)
        )

    train_frame = take(train_ids, "train")
    if train_frame.is_empty():
        raise ValueError("CARE training window has no rows before event onset")
    validation_split: Optional[TabularSplit]
    if validation_ids:
        validation_frame = take(validation_ids, "train")
        validation_split = TabularSplit(
            frame=validation_frame,
            feature_columns=feature_columns,
            target_column=TIME_TO_EVENT_COLUMN,
            entity_column=ENTITY_COLUMN,
            time_column=TIME_COLUMN,
        )
    else:
        validation_split = None
    test_labeled = take([*train_ids, *validation_ids], "prediction")
    if test_labeled.is_empty():
        raise ValueError(
            "CARE prediction window has no rows before event onset"
        )
    test_time = test_labeled.select(
        ENTITY_COLUMN,
        TIME_COLUMN,
        TIME_TO_EVENT_COLUMN,
        EVENT_OBSERVED_COLUMN,
    )
    test = TabularSplit(
        frame=test_labeled.select(
            [ENTITY_COLUMN, TIME_COLUMN, *feature_columns]
        ),
        feature_columns=feature_columns,
        target_column=None,
        entity_column=ENTITY_COLUMN,
        time_column=TIME_COLUMN,
    )
    train = TabularSplit(
        frame=train_frame,
        feature_columns=feature_columns,
        target_column=TIME_TO_EVENT_COLUMN,
        entity_column=ENTITY_COLUMN,
        time_column=TIME_COLUMN,
    )
    return train, validation_split, test, test_time


__all__ = [
    "CareTteBundle",
    "eval_care",
    "prepare_care",
]
