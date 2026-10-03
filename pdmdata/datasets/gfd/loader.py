"""GFD source layout and dataset-specific loading."""

from __future__ import annotations
import polars as pl
from pdmdata.io import find_raw_file

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 5)]


def load(condition: str = "healthy", load: int = 0) -> pl.DataFrame:
    labels = {"healthy": "h", "broken": "b"}
    if condition not in labels:
        raise ValueError("condition must be healthy or broken")
    if load not in range(0, 100, 10):
        raise ValueError("load must be between 0 and 90 in increments of 10")
    filename = "{}30hz{}.txt".format(labels[condition], load)
    path = find_raw_file("gfd", filename)
    # Source files start with blank formatting lines, before any measurements.
    leading_blanks = 0
    with path.open() as source:
        for line in source:
            if line.strip():
                break
            leading_blanks += 1
    frame = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        skip_rows=leading_blanks,
    )
    frame = frame.select(
        column
        for column in frame.columns
        if frame.get_column(column).null_count() < frame.height
    )
    # The distributed text files also end with blank formatting lines.
    while frame.height and frame.tail(1).null_count().row(0) == (1,) * frame.width:
        frame = frame.head(frame.height - 1)
    if frame.width != 4 or frame.is_empty():
        raise ValueError(f"Expected four nonempty vibration channels in {filename}")
    frame.columns = SENSOR_COLUMNS
    frame = frame.cast(pl.Float64)
    if frame.null_count().sum_horizontal().item() or not frame.select(
        pl.all().is_finite().all()
    ).row(0) == (True,) * 4:
        raise ValueError(f"Missing or non-finite measurements in {filename}")
    return frame.with_columns(
        pl.lit(condition).cast(pl.Categorical).alias("condition"),
        pl.lit(load).cast(pl.UInt8).alias("load"),
    )


def inventory() -> pl.DataFrame:
    """Read and validate all 20 local recordings and report their sample counts.

    No data is downloaded. Missing recordings or malformed measurements raise
    an error rather than returning an apparently complete inventory.
    """
    records = []
    for condition, prefix in (("healthy", "h"), ("broken", "b")):
        for level in range(0, 100, 10):
            filename = f"{prefix}30hz{level}.txt"
            path = find_raw_file("gfd", filename)
            frame = load(condition=condition, load=level)
            records.append({"condition": condition, "load": level,
                            "filename": filename, "samples": frame.height,
                            "bytes": path.stat().st_size, "path": str(path)})
    return pl.DataFrame(records)
