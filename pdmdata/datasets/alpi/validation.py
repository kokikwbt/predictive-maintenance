"""Inspect raw ALPI events while retaining ties and duplicates as observations."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from .loader import SOURCE_COLUMNS, load


def validate() -> dict:
    """Validate source fields and report source-order and duplicate diagnostics."""
    frame = load(lazy=False)
    if frame.is_empty():
        raise ValueError("ALPI contains no recorded events")
    if any(frame.select(SOURCE_COLUMNS).null_count().row(0)):
        raise ValueError("ALPI timestamp, alarm, and serial must be non-null")
    if frame.filter((pl.col("alarm") < 0) | (pl.col("serial") < 0)).height:
        raise ValueError("ALPI alarm and serial must be nonnegative source identifiers")
    gaps = frame.select(pl.col("timestamp").diff().over("serial").alias("gap"))["gap"]
    return {
        "records": frame.height,
        "machines": frame["serial"].n_unique(),
        "alarm_codes": frame["alarm"].n_unique(),
        "serial_ids": sorted(frame["serial"].unique().to_list()),
        "alarm_min": frame["alarm"].min(),
        "alarm_max": frame["alarm"].max(),
        "duplicates": frame.height - frame.select(SOURCE_COLUMNS).unique().height,
        "adjacent_ties_per_machine": (gaps == timedelta(0)).sum(),
        "out_of_order_per_machine": (gaps < timedelta(0)).sum(),
        "first_timestamp": frame["timestamp"].min().isoformat(),
        "last_timestamp": frame["timestamp"].max().isoformat(),
        "timezone": "unspecified by provider",
        "duplicates_removed": False,
    }
