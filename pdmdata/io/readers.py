"""Shared Polars-based readers for downloaded source data."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import polars as pl


def read_whitespace(
    path: Path | str,
    columns: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Read a whitespace-delimited text matrix as a Polars DataFrame."""
    frame = pl.read_csv(
        path,
        separator=" ",
        has_header=False,
        truncate_ragged_lines=True,
    )
    frame = frame.select(
        column
        for column in frame.columns
        if frame.get_column(column).null_count() < frame.height
    )
    if columns is not None:
        names = list(columns)
        if frame.width != len(names):
            raise ValueError(
                f"Expected {len(names)} columns in {path}, found {frame.width}"
            )
        frame.columns = names
    return frame
