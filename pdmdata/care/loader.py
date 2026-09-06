"""CARE's farm/event layout, semicolon CSVs, and train/prediction splits."""

from __future__ import annotations

import polars as pl

from ..io import find_raw_file


def load(
    recording: str | None = None,
    lazy: bool = True,
    *,
    wind_farm: str | None = None,
    event_id: int | None = None,
    table: str = "data",
    split: str | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Load one CARE event or a farm's supporting table without changing raw files.

    Select data with wind_farm="A", event_id=0. Use table="events" or
    table="features" for farm metadata. split may be "train" or "prediction".
    The previous recording="Wind Farm A/datasets/0.csv" API is also supported.
    """
    if table not in {"data", "events", "features"}:
        raise ValueError("table must be data, events, or features")
    if split is not None and split not in {"train", "prediction"}:
        raise ValueError("split must be train or prediction")

    if recording is not None:
        if wind_farm is not None or event_id is not None or table != "data" or split is not None:
            raise ValueError("Use recording alone, or select wind_farm/event_id/table/split")
        path = find_raw_file("care", recording)
    else:
        if not isinstance(wind_farm, str) or wind_farm.upper() not in {"A", "B", "C"}:
            raise ValueError("wind_farm must be A, B, or C")
        farm = wind_farm.upper()
        if table == "data":
            if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id < 0:
                raise ValueError("event_id must be a non-negative integer for table='data'")
            filename = f"datasets/{event_id}.csv"
        else:
            if event_id is not None or split is not None:
                raise ValueError("event_id and split apply only to table='data'")
            filename = "event_info.csv" if table == "events" else "feature_description.csv"
        path = find_raw_file("care", f"Wind Farm {farm}/{filename}")

    scan = pl.scan_csv(path, separator=";", try_parse_dates=True)
    if table == "events":
        # Farm A calls this column 'asset'; present the same name for every farm.
        if "asset" in scan.collect_schema().names():
            scan = scan.rename({"asset": "asset_id"})
    if split is not None:
        scan = scan.filter(pl.col("train_test") == split)
    return scan if lazy else scan.collect()
