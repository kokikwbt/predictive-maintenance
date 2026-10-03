"""METROPT2 source layout and dataset-specific loading."""

from __future__ import annotations
import polars as pl
from pdmdata.io import find_raw_file


def load(lazy: bool = True) -> pl.DataFrame | pl.LazyFrame:
    scan = pl.scan_csv(
        find_raw_file("metropt2", "MetroPT2.csv"),
        try_parse_dates=True,
    )
    return scan if lazy else scan.collect()
