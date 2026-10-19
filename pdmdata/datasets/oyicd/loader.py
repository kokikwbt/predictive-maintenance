"""OYICD recording selection, duplicate handling, and local validation."""

from __future__ import annotations

import re

import polars as pl

from pdmdata.config import get_settings

SENSOR_COLUMNS = [
    "pCut::Motor_Torque",
    "pCut::CTRL_Position_controller::Lag_error",
    "pCut::CTRL_Position_controller::Actual_position",
    "pCut::CTRL_Position_controller::Actual_speed",
    "pSvolFilm::CTRL_Position_controller::Actual_position",
    "pSvolFilm::CTRL_Position_controller::Actual_speed",
    "pSvolFilm::CTRL_Position_controller::Lag_error",
    "pSpintor::VAX_speed",
]
_PATTERN = re.compile(r"(0[1-9]|1[0-2])-([0-2][0-9]|3[01])T([0-2][0-9][0-5][0-9][0-5][0-9])_(\d+)_mode([1-8])\.csv")

_INVENTORY_SCHEMA = {
    "filename": pl.String, "month": pl.Int64, "mode": pl.Int64,
    "samples": pl.Int64, "copies": pl.Int64, "span_s": pl.Float64,
    "median_step_s": pl.Float64, "min_step_s": pl.Float64,
    "max_step_s": pl.Float64, "bytes": pl.Int64, "path": pl.String,
}


def _metadata(recording):
    match = _PATTERN.fullmatch(recording)
    if not match or int(match[2]) == 0 or int(match[3][:2]) > 23:
        raise ValueError("Use an OYICD filename: MM-DDTHHMMSS_NUM_modeX.csv")
    return {"filename": recording, "month": int(match[1]), "mode": int(match[5])}


def _resolve(paths):
    paths = sorted(paths, key=lambda path: (len(path.parts), str(path)))
    original = paths[0].read_bytes()
    if any(path.read_bytes() != original for path in paths[1:]):
        raise ValueError(f"Conflicting copies of OYICD recording {paths[0].name}")
    return paths[0]


def _find_recording(recording):
    paths = list((get_settings().data_root / "oyicd").rglob(recording))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise FileNotFoundError(f"OYICD recording {recording!r} is missing. Run pdmdata.download('oyicd') explicitly first.")
    return _resolve(paths)


def _read(path):
    frame = pl.read_csv(path)
    if frame.columns != ["timestamp", *SENSOR_COLUMNS] or frame.height < 2:
        raise ValueError(f"Expected timestamp and eight process signals in {path.name}")
    frame = frame.cast(pl.Float64)
    if frame.null_count().sum_horizontal().item() or not all(
        frame.select(pl.all().is_finite().all()).row(0)
    ):
        raise ValueError(f"Missing or non-finite measurements in {path.name}")
    if frame["timestamp"].diff().drop_nulls().min() <= 0:
        raise ValueError(f"Timestamps must increase strictly within {path.name}")
    return frame


def load(recording: str = "01-04T184148_000_mode1.csv") -> pl.DataFrame:
    """Load one recording, accepting only byte-identical duplicate source copies."""
    metadata = _metadata(recording)
    return _read(_find_recording(recording)).with_columns(
        pl.lit(recording).alias("filename"),
        pl.lit(metadata["month"]).cast(pl.UInt8).alias("month"),
        pl.lit(metadata["mode"]).cast(pl.UInt8).alias("mode"),
    )


def inventory(*, month: int | None = None, mode: int | None = None) -> pl.DataFrame:
    """Check local recordings and list each unique filename once.

    Optional month/mode selectors filter the inventory. Duplicate copies must
    agree byte for byte; no source files are changed or downloaded.
    """
    if month is not None and month not in range(1, 13):
        raise ValueError("month must be between 1 and 12")
    if mode is not None and mode not in range(1, 9):
        raise ValueError("mode must be between 1 and 8")
    groups = {}
    for path in (get_settings().data_root / "oyicd").rglob("*.csv"):
        if path.is_file():
            groups.setdefault(path.name, []).append(path)
    if not groups:
        raise FileNotFoundError("No local OYICD recordings. Run pdmdata.download('oyicd') explicitly first.")
    rows = []
    for filename, paths in sorted(groups.items()):
        metadata = _metadata(filename)
        if month is not None and metadata["month"] != month:
            continue
        if mode is not None and metadata["mode"] != mode:
            continue
        path = _resolve(paths)
        frame = _read(path)
        delta = frame["timestamp"].diff().drop_nulls()
        rows.append({**metadata, "samples": frame.height, "copies": len(paths),
                     "span_s": frame["timestamp"][-1] - frame["timestamp"][0],
                     "median_step_s": delta.median(), "min_step_s": delta.min(),
                     "max_step_s": delta.max(), "bytes": path.stat().st_size,
                     "path": str(path)})
    return pl.DataFrame(rows, schema=_INVENTORY_SCHEMA)
