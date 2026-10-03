"""MAPM source layout and dataset-specific loading."""

from __future__ import annotations
import polars as pl
from pdmdata.io import find_raw_file


MAPM_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
    "maintenance": "PdM_maint.csv",
}


def load(table: str = "telemetry") -> pl.DataFrame:
    try:
        filename = MAPM_FILES[table]
    except KeyError:
        raise ValueError(
            "table must be one of: {}".format(", ".join(sorted(MAPM_FILES)))
        ) from None
    return pl.read_csv(find_raw_file("mapm", filename), try_parse_dates=True)
