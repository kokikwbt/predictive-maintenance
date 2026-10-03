"""PPD source layout and dataset-specific loading."""

from __future__ import annotations
import polars as pl
from pdmdata.io import find_raw_file


SEQUENCE_FILES = {
    7: ("C7-1", "C7-2"),
    8: ("C8",),
    9: ("C9",),
    11: ("C11",),
    13: ("C13-1", "C13-2"),
    14: ("C14",),
    15: ("C15",),
    16: ("C16",),
}
SEQUENCE_IDS = tuple(SEQUENCE_FILES)
PPD_FILES = {name for parts in SEQUENCE_FILES.values() for name in parts}


def _sequence_id(value):
    if isinstance(value, str):
        value = value.removeprefix("C")
        if value in {str(i) for i in SEQUENCE_IDS}:
            return int(value)
    elif type(value) is int and value in SEQUENCE_FILES:
        return value
    raise ValueError(f"sequence_id must be one of {SEQUENCE_IDS}, optionally prefixed with C")


SENSOR_COLUMNS = ([f"L_{i}" for i in range(1, 11)]
                  + [f"{group}_{i}" for group in "ABC" for i in range(1, 6)])


def load_file(experiment: str = "C7-1") -> pl.DataFrame:
    """Load one file, preserving source order, values, and missing measurements.

    Experiment IDs identify recordings, not health-state labels. Timestamp is
    a source index; its physical time unit is not established by the CSV.
    """
    if experiment not in PPD_FILES:
        raise ValueError(
            "experiment must be one of: {}".format(", ".join(sorted(PPD_FILES)))
        )
    frame = pl.read_csv(find_raw_file("ppd", f"{experiment}.csv"))
    if set(frame.columns) != {"Timestamp", *SENSOR_COLUMNS} or frame.is_empty():
        raise ValueError("Expected Timestamp and 25 process signals in a nonempty PPD recording")
    if not frame["Timestamp"].dtype.is_integer() or frame["Timestamp"].null_count():
        raise ValueError("PPD Timestamp must contain non-null integer indices")
    if frame.height > 1 and frame["Timestamp"].diff().drop_nulls().min() <= 0:
        raise ValueError("PPD Timestamp indices must increase within each file")
    frame = frame.with_columns(pl.col(SENSOR_COLUMNS).cast(pl.Float64))
    if not all(frame.select(pl.col(SENSOR_COLUMNS).is_finite().fill_null(True).all()).row(0)):
        raise ValueError("PPD signals contain non-finite values; source nulls are allowed")
    sequence_id = int(experiment.split("-")[0][1:])
    return frame.with_columns(
        pl.lit(sequence_id).cast(pl.UInt8).alias("sequence_id"),
        pl.lit(f"{experiment}.csv").alias("source_file"),
    )



def load(sequence_id: int | str = 7) -> pl.DataFrame:
    """Load one complete sequence, joining its source parts in numeric order.

    The zero-based sample column is continuous across parts. Timestamp retains
    the original per-file index, including resets; source_file preserves each
    row's provenance. Missing rows remain in place and no gap duration is inferred.
    """
    sequence_id = _sequence_id(sequence_id)
    parts = [load_file(name) for name in SEQUENCE_FILES[sequence_id]]
    # Select a common column order before joining source files.
    parts = [frame.select(parts[0].columns) for frame in parts]
    return pl.concat(parts).with_row_index("sample")


def inventory() -> pl.DataFrame:
    """Validate all eight sequences, including every required source part."""
    rows = []
    for sequence_id, parts in SEQUENCE_FILES.items():
        frame = load(sequence_id)
        paths = [find_raw_file("ppd", f"{part}.csv") for part in parts]
        signals = frame.select(SENSOR_COLUMNS)
        rows.append({"sequence_id": sequence_id, "parts": len(parts),
                     "source_files": [path.name for path in paths],
                     "samples": frame.height, "first_sample": 0,
                     "last_sample": frame.height - 1,
                     "missing_values": signals.null_count().sum_horizontal().item(),
                     "rows_with_missing": signals.select(pl.any_horizontal(pl.all().is_null()).sum()).item(),
                     "constant_signals": [c for c in SENSOR_COLUMNS if signals[c].drop_nulls().n_unique() <= 1],
                     "bytes": sum(path.stat().st_size for path in paths)})
    return pl.DataFrame(rows)
