"""GDD source layout and dataset-specific loading."""

from __future__ import annotations
import polars as pl
from pdmdata.io import find_raw_file


GDD_FILES = {
    "state": "Genesis_StateMachineLabel.csv",
    "anomaly": "Genesis_AnomalyLabels.csv",
    "normal": "Genesis_normal.csv",
    "linear": "Genesis_lineardrive.csv",
    "pressure": "Genesis_pressure.csv",
}


def load(series: str = "state") -> pl.DataFrame:
    try:
        filename = GDD_FILES[series]
    except KeyError:
        raise ValueError(
            "series must be one of: {}".format(", ".join(sorted(GDD_FILES)))
        ) from None
    return pl.read_csv(find_raw_file("gdd", filename), try_parse_dates=True)
