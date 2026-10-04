"""SCANIA Component X source layout, typed loading, and label helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from pdmdata.io import find_raw_file


DATASET_ID = "scania_x"
SPLITS = ("train", "validation", "test")
TABLES = ("readouts", "specifications", "tte", "labels")
COUNTERS = (
    "171_0", "666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0",
)
HISTOGRAMS = {"167": 10, "272": 10, "291": 11, "158": 10, "459": 20, "397": 36}
SPECIFICATIONS = tuple("Spec_{}".format(index) for index in range(8))
# Upper bounds of the provider's time windows before repair, for classes 4..1.
# The provider does not state boundary inclusion; windows are half-open here.
WINDOW_BOUNDS = ((4, 6.0), (3, 12.0), (2, 24.0), (1, 48.0))


def histogram_columns(variable: str) -> list[str]:
    """Return the ordered bin columns of one anonymized histogram variable."""
    try:
        bins = HISTOGRAMS[variable]
    except KeyError:
        raise ValueError(
            "variable must be one of: {}".format(", ".join(HISTOGRAMS))
        ) from None
    return ["{}_{}".format(variable, index) for index in range(bins)]


FEATURES = tuple(COUNTERS) + tuple(
    column for variable in HISTOGRAMS for column in histogram_columns(variable)
)


def _filename(split: str, table: str) -> str:
    if split not in SPLITS:
        raise ValueError("split must be one of: {}".format(", ".join(SPLITS)))
    if table not in TABLES:
        raise ValueError("table must be one of: {}".format(", ".join(TABLES)))
    if table == "tte" and split != "train":
        raise ValueError("table='tte' is only available for split='train'")
    if table == "labels" and split == "train":
        raise ValueError(
            "table='labels' is only available for validation and test; "
            "use table='tte' for train"
        )
    stem = "operational_readouts" if table == "readouts" else table
    return "{}_{}.csv".format(split, stem)


def _path(split: str, table: str) -> Path:
    return find_raw_file(DATASET_ID, _filename(split, table))


def _scan_readouts(path: Path) -> pl.LazyFrame:
    # Read as strings first: a sparse all-null prefix must not decide a type.
    scan = pl.scan_csv(path, infer_schema=False)
    names = scan.collect_schema().names()
    expected = ["vehicle_id", "time_step", *FEATURES]
    if sorted(names) != sorted(expected):
        raise ValueError("Unexpected readout columns in {}".format(path))
    return scan.select(
        pl.col(name).cast(pl.Int64 if name == "vehicle_id" else pl.Float64)
        for name in names
    )


def _read_table(path: Path, table: str) -> pl.DataFrame:
    schema = {"vehicle_id": pl.Int64}
    if table == "specifications":
        schema.update({name: pl.Utf8 for name in SPECIFICATIONS})
    elif table == "tte":
        schema.update(
            length_of_study_time_step=pl.Float64, in_study_repair=pl.UInt8
        )
    else:
        schema.update(class_label=pl.UInt8)
    frame = pl.read_csv(path, schema_overrides=schema)
    if set(frame.columns) != set(schema):
        raise ValueError("Unexpected {} columns in {}".format(table, path))
    return frame.select(list(schema))


def _vehicle_filter(vehicle_id: int | Sequence[int] | None) -> pl.Expr | None:
    if vehicle_id is None:
        return None
    if isinstance(vehicle_id, bool):
        raise ValueError("vehicle_id must be an integer or a list of integers")
    if isinstance(vehicle_id, int):
        return pl.col("vehicle_id") == vehicle_id
    ids = list(vehicle_id)
    if not ids or any(
        isinstance(item, bool) or not isinstance(item, int) for item in ids
    ):
        raise ValueError("vehicle_id must be an integer or a list of integers")
    return pl.col("vehicle_id").is_in(ids)


def load(
    split: str = "train",
    table: str = "readouts",
    lazy: bool | None = None,
    *,
    vehicle_id: int | Sequence[int] | None = None,
    columns: list[str] | None = None,
    with_labels: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    """Load one source table for a split without changing its values.

    ``table`` is ``readouts`` (operational time series), ``specifications``,
    ``tte`` (train repair records), or ``labels`` (validation/test classes).
    Readouts are lazy by default; the small per-vehicle tables are eager.
    ``with_labels=True`` adds derived train targets (see ``add_tte_targets``).
    """
    path = _path(split, table)
    if table == "readouts":
        frame = _scan_readouts(path)
    else:
        frame = _read_table(path, table).lazy()
    if with_labels:
        if (split, table) != ("train", "readouts"):
            raise ValueError(
                "with_labels is only supported for train readouts; use "
                "last_readouts() for validation and test labels"
            )
        frame = add_tte_targets(frame, load("train", "tte"))
    selection = _vehicle_filter(vehicle_id)
    if selection is not None:
        frame = frame.filter(selection)
    if columns is not None:
        if not isinstance(columns, list) or not columns or any(
            not isinstance(column, str) for column in columns
        ):
            raise ValueError("columns must be a non-empty list of column names")
        missing = set(columns) - set(frame.collect_schema().names())
        if missing:
            raise ValueError("Unknown columns: " + ", ".join(sorted(missing)))
        frame = frame.select(list(dict.fromkeys(columns)))
    if lazy is None:
        lazy = table == "readouts"
    return frame if lazy else frame.collect()


def add_tte_targets(
    readouts: pl.DataFrame | pl.LazyFrame, tte: pl.DataFrame
) -> pl.DataFrame | pl.LazyFrame:
    """Join train repair records and derive per-readout targets.

    Adds ``time_to_event`` (study end minus ``time_step``), ``event_observed``
    (``in_study_repair``), and ``class_label`` using the provider's windows.
    Censored readouts closer than 48 time steps to the end of observation
    have an unknown class and receive a null label.
    """
    lazy = isinstance(readouts, pl.LazyFrame)
    records = tte.lazy().select(
        "vehicle_id",
        "length_of_study_time_step",
        pl.col("in_study_repair").alias("event_observed"),
    )
    remaining = pl.col("length_of_study_time_step") - pl.col("time_step")
    label = pl.when(remaining >= WINDOW_BOUNDS[-1][1]).then(pl.lit(0))
    for value, upper in WINDOW_BOUNDS:
        label = label.when(
            (pl.col("event_observed") == 1) & (remaining < upper)
        ).then(pl.lit(value))
    frame = (
        readouts.lazy()
        .join(records, on="vehicle_id", how="left", validate="m:1")
        .with_columns(
            remaining.alias("time_to_event"),
            label.otherwise(None).cast(pl.UInt8).alias("class_label"),
        )
        .drop("length_of_study_time_step")
    )
    return frame if lazy else frame.collect()


def vehicles(split: str = "train") -> pl.DataFrame:
    """Return one row per vehicle: specifications plus TTE or class labels."""
    targets = load(split, "tte" if split == "train" else "labels")
    frame = load(split, "specifications").join(
        targets, on="vehicle_id", how="left", validate="1:1"
    )
    return frame.sort("vehicle_id")


def last_readouts(
    split: str = "validation", lazy: bool = False
) -> pl.DataFrame | pl.LazyFrame:
    """Return each vehicle's last readout joined with its specifications.

    For validation and test this is the prediction point of the IDA 2024
    challenge, and ``class_label`` is attached. For train, the repair records
    are attached instead.
    """
    readouts = load(split, "readouts")
    last = (
        readouts.sort(["vehicle_id", "time_step"], maintain_order=True)
        .group_by("vehicle_id", maintain_order=True)
        .last()
    )
    frame = last.join(
        vehicles(split).lazy(), on="vehicle_id", how="left", validate="1:1"
    )
    return frame if lazy else frame.collect()


def inventory() -> pl.DataFrame:
    """List local source files and byte sizes without reading their rows."""
    rows = []
    for split in SPLITS:
        for table in TABLES:
            try:
                name = _filename(split, table)
            except ValueError:
                continue
            path = _path(split, table)
            rows.append({
                "split": split,
                "table": table,
                "file": name,
                "bytes": path.stat().st_size,
            })
    return pl.DataFrame(rows)
