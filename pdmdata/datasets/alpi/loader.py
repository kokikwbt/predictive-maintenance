"""Load the original ALPI alarm events without constructing prediction targets."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import polars as pl

from pdmdata.io import find_raw_file

SOURCE_COLUMNS = ("timestamp", "alarm", "serial")
COLUMNS = ("source_row", *SOURCE_COLUMNS)


def _ids(value: int | Sequence[int] | None, name: str) -> list[int] | None:
    if value is None:
        return None
    if type(value) is int:
        values = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise ValueError(f"{name} must be a nonnegative integer or a nonempty sequence of integers")
    if not values or any(type(item) is not int or item < 0 for item in values):
        raise ValueError(f"{name} must be a nonnegative integer or a nonempty sequence of integers")
    return values


def _time(value: datetime | str | None, name: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise ValueError(f"{name} must be an ISO-format datetime") from None
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError(f"{name} must be a timezone-naive datetime; ALPI does not specify a timezone")
    return value


def load(
    *,
    serial: int | Sequence[int] | None = None,
    alarm: int | Sequence[int] | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
    columns: Sequence[str] | None = None,
    lazy: bool = True,
) -> pl.LazyFrame | pl.DataFrame:
    """Read ``alarms.csv`` with optional machine, alarm, and [start, end) filters.

    Original IDs, duplicate events, source order, and timestamp precision are
    preserved. ``source_row`` is the zero-based original CSV row, excluding the
    header. Alarm ID 0 is a real raw-data code, not padding. Timezone is unknown.
    Loading never downloads data or opens the provider's pickle/NPZ artifacts.
    """
    serials = _ids(serial, "serial")
    alarms = _ids(alarm, "alarm")
    start_time, end_time = _time(start, "start"), _time(end, "end")
    if start_time is not None and end_time is not None and start_time >= end_time:
        raise ValueError("start must be earlier than end")
    if columns is not None:
        if (isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence)
                or not columns or any(not isinstance(c, str) or c not in COLUMNS for c in columns)
                or len(set(columns)) != len(columns)):
            raise ValueError(f"columns must be a nonempty selection from {COLUMNS} without duplicates")
    if type(lazy) is not bool:
        raise ValueError("lazy must be True or False")
    path = find_raw_file("alpi", "alarms.csv")
    scan = pl.scan_csv(
        path,
        schema_overrides={"timestamp": pl.String, "alarm": pl.Int64, "serial": pl.Int64},
        row_index_name="source_row",
    )
    if scan.collect_schema().names() != list(COLUMNS):
        raise ValueError("ALPI requires exactly timestamp, alarm, and serial CSV columns in source order")
    frame = scan.with_columns(
        pl.col("timestamp").str.to_datetime(
            format="%Y-%m-%d %H:%M:%S%.f", time_unit="us", strict=True
        )
    )
    if serials is not None:
        frame = frame.filter(pl.col("serial").is_in(serials))
    if alarms is not None:
        frame = frame.filter(pl.col("alarm").is_in(alarms))
    if start_time is not None:
        frame = frame.filter(pl.col("timestamp") >= start_time)
    if end_time is not None:
        frame = frame.filter(pl.col("timestamp") < end_time)
    if columns is not None:
        frame = frame.select(list(columns))
    return frame if lazy else frame.collect()


def inventory() -> pl.DataFrame:
    """Summarize recorded events per machine; last alarms are not censoring dates."""
    frame = load(lazy=False)
    return frame.group_by("serial").agg(
        pl.len().alias("events"),
        pl.col("alarm").n_unique().alias("alarm_codes"),
        pl.col("timestamp").min().alias("first_timestamp"),
        pl.col("timestamp").max().alias("last_timestamp"),
    ).sort("serial")
