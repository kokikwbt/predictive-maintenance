"""Describe contiguous GDD states without reordering source observations."""

import polars as pl


MOTOR_SIGNALS = [
    "MotorData.ActCurrent", "MotorData.ActPosition", "MotorData.ActSpeed",
    "MotorData.IsAcceleration", "MotorData.IsForce",
]


def state_segments(frame: pl.DataFrame) -> pl.DataFrame:
    """Return half-open sample intervals; IDs are source labels, not named actions."""
    if frame.is_empty() or frame["Label"].null_count():
        raise ValueError("State segments require non-empty, non-null labels")
    return (
        frame.select("Label").with_row_index("sample")
        .with_columns((pl.col("Label") != pl.col("Label").shift(1))
                      .fill_null(True).cast(pl.UInt32).cum_sum().alias("segment"))
        .group_by("segment", maintain_order=True)
        .agg(pl.col("Label").first(), pl.col("sample").min().alias("start_sample"),
             (pl.col("sample").max() + 1).alias("end_sample"), pl.len().alias("samples"))
    )
