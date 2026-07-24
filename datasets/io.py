"""Shared Polars-based readers for downloaded source data."""

from pathlib import Path
from typing import Iterable, Optional

import polars as pl

from .download import DEFAULT_DATA_ROOT


def find_raw_file(dataset_id: str, filename: str) -> Path:
    """Find a source file below ``data/raw/<dataset-id>``."""
    directory = DEFAULT_DATA_ROOT / dataset_id
    matches = [path for path in directory.rglob(filename) if path.is_file()]
    if not matches:
        raise FileNotFoundError(
            "{} was not found under {}. Download or place the dataset first.".format(
                filename, directory
            )
        )
    if len(matches) > 1:
        raise ValueError(
            "Multiple files named {} were found under {}".format(filename, directory)
        )
    return matches[0]


def read_whitespace(
    path: Path | str,
    columns: Optional[Iterable[str]] = None,
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
                "Expected {} columns in {}, found {}".format(
                    len(names), path, frame.width
                )
            )
        frame.columns = names
    return frame
