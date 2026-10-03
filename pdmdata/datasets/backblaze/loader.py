"""Select local Backblaze snapshots and normalize sparse quarterly schemas."""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import polars as pl

from pdmdata.catalog import load_metadata
from pdmdata.config import get_settings


def _variant(variant: str | None) -> str:
    download = load_metadata("backblaze")["download"]
    variant = download["default_variant"] if variant is None else variant
    if variant not in download["variants"]:
        raise ValueError("variant must be one of: " + ", ".join(download["variants"]))
    return variant


def _date(value: str | date | None) -> date | None:
    if value is None:
        return None
    if type(value) is date:
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("Dates must be ISO YYYY-MM-DD strings or date objects")


def _files(variant: str, start: date | None = None, end: date | None = None) -> list[Path]:
    directory = get_settings().data_root / "backblaze" / variant / "extracted"
    files = []
    seen = set()
    for path in sorted(directory.rglob("*.csv")):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}\.csv", path.name) or "__MACOSX" in path.parts:
            continue
        day = date.fromisoformat(path.stem)
        if (start and day < start) or (end and day > end):
            continue
        if day in seen:
            raise ValueError(f"Duplicate daily snapshot: {day}")
        seen.add(day)
        files.append(path)
    if not files:
        raise FileNotFoundError(
            f"No Backblaze daily CSVs match the selection under {directory}. "
            f"Download variant {variant!r} explicitly first."
        )
    return files


def _scan(path: Path) -> pl.LazyFrame:
    # Read sparse fields as strings first: an all-null prefix must not decide
    # the type of a SMART column for an entire daily file.
    scan = pl.scan_csv(path, infer_schema=False)
    names = scan.collect_schema().names()
    types = {"date": pl.Date, "failure": pl.UInt8, "capacity_bytes": pl.Int64}
    return scan.with_columns(
        pl.col(name).str.to_date("%Y-%m-%d") if name == "date" else
        pl.col(name).cast(pl.Float64 if name.startswith("smart_") else types[name])
        for name in names if name.startswith("smart_") or name in types
    )


def load(
    variant: str | None = None,
    lazy: bool = True,
    *,
    serial_number: str | None = None,
    model: str | None = None,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    columns: list[str] | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Load a quarter or selected drive history; date bounds are inclusive.

    Missing SMART fields across daily schemas become null, never zero.
    Source files are not changed. Rows retain daily file order.
    """
    variant = _variant(variant)
    start, end = _date(start_date), _date(end_date)
    if start and end and start > end:
        raise ValueError("start_date must not be after end_date")
    for name, value in (("serial_number", serial_number), ("model", model)):
        if value is not None and (not isinstance(value, str) or not value):
            raise ValueError(f"{name} must be a non-empty string")
    if columns is not None and (not isinstance(columns, list) or not columns or
                               any(not isinstance(c, str) for c in columns)):
        raise ValueError("columns must be a non-empty list of column names")
    scan = pl.concat([_scan(p) for p in _files(variant, start, end)], how="diagonal_relaxed")
    for name, value in (("serial_number", serial_number), ("model", model)):
        if value is not None:
            scan = scan.filter(pl.col(name) == value)
    if start:
        scan = scan.filter(pl.col("date") >= start)
    if end:
        scan = scan.filter(pl.col("date") <= end)
    if columns is not None:
        missing = set(columns) - set(scan.collect_schema().names())
        if missing:
            raise ValueError("Unknown columns: " + ", ".join(sorted(missing)))
        scan = scan.select(list(dict.fromkeys(columns)))
    return scan if lazy else scan.collect()


def inventory(variant: str | None = None) -> pl.DataFrame:
    """List local daily files, dates, and byte sizes without reading all rows."""
    variant = _variant(variant)
    return pl.DataFrame([
        {"date": date.fromisoformat(p.stem), "path": str(p), "bytes": p.stat().st_size}
        for p in _files(variant)
    ])
