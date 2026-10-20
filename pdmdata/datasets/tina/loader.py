"""Source-faithful daily TiNA Parquet loading; no event targets or cleaning."""
from __future__ import annotations
from collections.abc import Sequence
from datetime import date
from pathlib import Path
import polars as pl
from pdmdata.config import get_settings


def _days(value: str | Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value) if isinstance(value, Sequence) else []
    if not values or any(not isinstance(item, str) for item in values) or len(set(values)) != len(values):
        raise ValueError("days must be an ISO date or a nonempty sequence of distinct ISO dates")
    for item in values:
        if not isinstance(item, str):
            raise ValueError("Every day must be an ISO date string")
        try:
            parsed = date.fromisoformat(item)
        except ValueError:
            raise ValueError("Every day must have YYYY-MM-DD format") from None
        if parsed.isoformat() != item:
            raise ValueError("Every day must have YYYY-MM-DD format")
    return sorted(values)


def load(*, days: str | Sequence[str] | None = None,
         columns: Sequence[str] | None = None, lazy: bool = True) -> pl.LazyFrame | pl.DataFrame:
    """Read installed raw day files, preserving values and within-file order.

    No missing days are downloaded or filled. No alarm onset, timestamp meaning,
    failure target, or sampling interval is inferred. source_row is zero-based
    within the original file; source_file is relative to the dataset directory.
    """
    selected = _days(days)
    if type(lazy) is not bool:
        raise ValueError("lazy must be True or False")
    if columns is not None and (isinstance(columns, (str, bytes)) or not isinstance(columns, Sequence)
                               or not columns or any(not isinstance(c, str) for c in columns)
                               or len(set(columns)) != len(columns)):
        raise ValueError("columns must be a nonempty sequence of distinct column names")
    root = get_settings().data_root / "tina"
    paths = ([root / ("day=" + item) / "data.parquet" for item in selected]
             if selected is not None else sorted(root.glob("day=????-??-??/data.parquet")))
    if not paths:
        raise FileNotFoundError("No TiNA day files found under " + str(root))
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError("TiNA source day is not installed: " + str(path))
        scan = pl.scan_parquet(path, hive_partitioning=False, row_index_name="source_row")
        if "source_file" in scan.collect_schema().names():
            raise ValueError("Raw TiNA file conflicts with source_file provenance")
        frames.append(scan.with_columns(pl.lit(str(path.relative_to(root))).alias("source_file")))
    frame = pl.concat(frames, how="vertical")
    if columns is not None:
        unknown = set(columns) - set(frame.collect_schema().names())
        if unknown:
            raise ValueError("Unknown TiNA columns: " + ", ".join(sorted(unknown)))
        frame = frame.select(list(columns))
    return frame if lazy else frame.collect()
