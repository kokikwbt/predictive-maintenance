"""C-MAPSS engine trajectories and explicit remaining-life targets."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import polars as pl

from pdmdata.io import find_raw_file

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SENSOR_COLUMNS = [f"sensor_{index}" for index in range(1, 22)]
OPERATION_COLUMNS = [f"operation_{index}" for index in range(1, 4)]
CMAPSS_COLUMNS = ["unit_number", "cycle", *OPERATION_COLUMNS, *SENSOR_COLUMNS]
CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
FAULT_MODES = {"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2}


def _subset(value: str) -> str:
    if value not in SUBSETS:
        raise ValueError("subset must be FD001, FD002, FD003, or FD004")
    return value


def _matrix(path: Path, width: int) -> np.ndarray:
    """Accept arbitrary whitespace but never discard extra or missing fields."""
    try:
        values = np.loadtxt(path, ndmin=2)
    except ValueError as error:
        raise ValueError(f"Invalid C-MAPSS matrix in {path}: {error}") from error
    if not values.size or values.shape[1] != width:
        raise ValueError(f"Expected a nonempty C-MAPSS matrix with {width} columns in {path}")
    if not np.isfinite(values).all():
        raise ValueError(f"Missing or non-finite C-MAPSS values in {path}")
    return values


def _series(subset: str, split: str) -> pl.DataFrame:
    values = _matrix(find_raw_file("cmapss", f"{split}_{subset}.txt"), 26)
    identifiers = values[:, :2]
    if (np.any(identifiers < 1) or np.any(identifiers != np.floor(identifiers))
            or np.any(values[:, 0] > 65535) or np.any(values[:, 1] > 4294967295)):
        raise ValueError("C-MAPSS unit numbers and cycles must be positive integers within their integer type bounds")
    frame = pl.DataFrame(values, schema=CMAPSS_COLUMNS, orient="row").with_columns(
        pl.col("unit_number").cast(pl.UInt16), pl.col("cycle").cast(pl.UInt32))
    units = frame["unit_number"].unique().sort().to_list()
    if units != list(range(1, len(units) + 1)):
        raise ValueError("C-MAPSS unit numbers must be consecutive starting at 1")
    expected = pl.col("cycle").cum_count().over("unit_number")
    if not frame.select((pl.col("cycle") == expected).all()).item():
        raise ValueError("C-MAPSS cycles must be consecutive starting at 1 in source order for each unit")
    return frame


def _rul(subset: str, expected_units: int) -> pl.DataFrame:
    values = _matrix(find_raw_file("cmapss", f"RUL_{subset}.txt"), 1)[:, 0]
    if (len(values) != expected_units or np.any(values < 0)
            or np.any(values != np.floor(values)) or np.any(values > 4294967295)):
        raise ValueError("C-MAPSS RUL must contain one nonnegative integer per test unit in unit-number order")
    return pl.DataFrame({"unit_number": range(1, expected_units + 1), "RUL": values},
                        schema={"unit_number": pl.UInt16, "RUL": pl.UInt32})


def rul(subset: str = "FD001") -> pl.DataFrame:
    """Return official test-end RUL, explicitly keyed by one-based unit number."""
    subset = _subset(subset)
    return _rul(subset, _series(subset, "test")["unit_number"].n_unique())


def load(subset: str = "FD001", split: str = "train", *,
         unit: int | None = None, with_rul: bool = False) -> pl.DataFrame:
    """Load a local subset/split, optionally selecting one engine or adding RUL.

    Default train/test output retains the 26 source columns. Train RUL is
    max(cycle) - cycle; test RUL additionally includes the official test-end
    offset. Test targets are opt-in evaluation data and must not enter features.
    The legacy ``split='rul'`` output retains its single RUL column.
    """
    subset = _subset(subset)
    if split not in {"train", "test", "rul"}:
        raise ValueError("split must be train, test, or rul")
    if unit is not None and (isinstance(unit, bool) or not isinstance(unit, int) or unit < 1):
        raise ValueError("unit must be a positive integer")
    if split == "rul":
        if with_rul:
            raise ValueError("with_rul is only used with train or test")
        frame = rul(subset)
    else:
        frame = _series(subset, split)
        if with_rul:
            target = pl.col("cycle").max().over("unit_number").cast(pl.Int64) - pl.col("cycle")
            if split == "test":
                offsets = _rul(subset, frame["unit_number"].n_unique())
                frame = frame.join(offsets, on="unit_number", how="left", validate="m:1", maintain_order="left")
                target = target + pl.col("RUL").cast(pl.Int64)
            frame = frame.with_columns(target.cast(pl.UInt32).alias("RUL"))
    if unit is not None:
        frame = frame.filter(pl.col("unit_number") == unit)
        if frame.is_empty():
            raise ValueError(f"No unit {unit} in {subset} {split}")
    return frame.select("RUL") if split == "rul" else frame


def inventory(subset: str | None = None) -> pl.DataFrame:
    """Validate files and summarize engines, cycles, and operating conditions."""
    selected = SUBSETS if subset is None else (_subset(subset),)
    rows = []
    for name in selected:
        for split in ("train", "test"):
            frame = _series(name, split)
            lengths = frame.group_by("unit_number").len()["len"]
            if split == "test":
                _rul(name, len(lengths))
            rows.append({"subset": name, "split": split, "units": len(lengths),
                         "rows": frame.height, "min_cycles": lengths.min(),
                         "max_cycles": lengths.max(), "conditions": CONDITIONS[name],
                         "fault_modes": FAULT_MODES[name]})
    return pl.DataFrame(rows)
