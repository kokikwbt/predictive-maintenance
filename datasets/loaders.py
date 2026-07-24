"""Internal Polars adapters for dataset-specific source layouts."""

from pathlib import Path
from typing import Any, Callable

import polars as pl

from .io import find_raw_file, read_whitespace


Loader = Callable[..., pl.DataFrame]

CMAPSS_COLUMNS = (
    ["unit_number", "cycle"]
    + [f"operation_{index}" for index in range(1, 4)]
    + [f"sensor_{index}" for index in range(1, 22)]
)
CBM_COLUMNS = [
    "lp",
    "v",
    "GTT",
    "GTn",
    "GGn",
    "Ts",
    "Tp",
    "T48",
    "T1",
    "T2",
    "P48",
    "P1",
    "P2",
    "Pexh",
    "TIC",
    "mf",
    "kMc",
    "kMt",
]
GDD_FILES = {
    "state": "Genesis_StateMachineLabel.csv",
    "anomaly": "Genesis_AnomalyLabels.csv",
    "normal": "Genesis_normal.csv",
    "linear": "Genesis_lineardrive.csv",
    "pressure": "Genesis_pressure.csv",
}
MAPM_FILES = {
    "telemetry": "PdM_telemetry.csv",
    "errors": "PdM_errors.csv",
    "failures": "PdM_failures.csv",
    "machines": "PdM_machines.csv",
    "maintenance": "PdM_maint.csv",
}
PPD_FILES = {
    "C7-1",
    "C7-2",
    "C8",
    "C9",
    "C11",
    "C13-1",
    "C13-2",
    "C14",
    "C15",
    "C16",
}


def load(dataset_id: str, **options: Any) -> pl.DataFrame:
    """Load a downloaded dataset as a Polars DataFrame."""
    try:
        adapter = _LOADERS[dataset_id]
    except KeyError:
        raise KeyError("Unknown dataset: {!r}".format(dataset_id)) from None
    try:
        return adapter(**options)
    except TypeError as error:
        raise TypeError("Invalid options for {!r}: {}".format(dataset_id, error)) from error


def _alpi() -> pl.DataFrame:
    return pl.read_csv(find_raw_file("alpi", "alarms.csv"), try_parse_dates=True)


def _cbm() -> pl.DataFrame:
    return read_whitespace(find_raw_file("cbm", "data.txt"), CBM_COLUMNS)


def _cmapss(subset: str = "FD001", split: str = "train") -> pl.DataFrame:
    if subset not in {"FD001", "FD002", "FD003", "FD004"}:
        raise ValueError("subset must be FD001, FD002, FD003, or FD004")
    if split == "rul":
        return read_whitespace(
            find_raw_file("cmapss", "RUL_{}.txt".format(subset)), ["RUL"]
        ).with_columns(pl.col("RUL").cast(pl.UInt32))
    if split not in {"train", "test"}:
        raise ValueError("split must be train, test, or rul")
    frame = read_whitespace(
        find_raw_file("cmapss", "{}_{}.txt".format(split, subset)),
        CMAPSS_COLUMNS,
    )
    return frame.with_columns(
        pl.col("unit_number").cast(pl.UInt16),
        pl.col("cycle").cast(pl.UInt32),
    )


def _gdd(series: str = "state") -> pl.DataFrame:
    try:
        filename = GDD_FILES[series]
    except KeyError:
        raise ValueError(
            "series must be one of: {}".format(", ".join(sorted(GDD_FILES)))
        ) from None
    return pl.read_csv(find_raw_file("gdd", filename), try_parse_dates=True)


def _gfd(condition: str = "healthy", load: int = 0) -> pl.DataFrame:
    labels = {"healthy": "h", "broken": "b"}
    if condition not in labels:
        raise ValueError("condition must be healthy or broken")
    if load not in range(0, 100, 10):
        raise ValueError("load must be between 0 and 90 in increments of 10")
    filename = "{}30hz{}.txt".format(labels[condition], load)
    return read_whitespace(
        find_raw_file("gfd", filename), ["sensor_1", "sensor_2", "sensor_3", "sensor_4"]
    ).with_columns(
        pl.lit(condition).cast(pl.Categorical).alias("condition"),
        pl.lit(load).cast(pl.UInt8).alias("load"),
    )


def _hydsys(sensor: str = "PS1") -> pl.DataFrame:
    filename = sensor if sensor.endswith(".txt") else "{}.txt".format(sensor)
    return read_whitespace(find_raw_file("hydsys", filename))


def _mapm(table: str = "telemetry") -> pl.DataFrame:
    try:
        filename = MAPM_FILES[table]
    except KeyError:
        raise ValueError(
            "table must be one of: {}".format(", ".join(sorted(MAPM_FILES)))
        ) from None
    return pl.read_csv(find_raw_file("mapm", filename), try_parse_dates=True)


def _ims(recording: str) -> pl.DataFrame:
    frame = pl.read_csv(
        find_raw_file("ims", recording),
        separator="\t",
        has_header=False,
    )
    frame.columns = [
        "channel_{}".format(index) for index in range(1, frame.width + 1)
    ]
    return frame.with_row_index("sample")


def _oyicd(recording: str = "01-04T184148_000_mode1.csv") -> pl.DataFrame:
    return pl.read_csv(
        find_raw_file("oyicd", recording),
        try_parse_dates=True,
    ).with_columns(
        pl.lit(recording).alias("filename"),
        pl.lit(int(recording[:2])).cast(pl.UInt8).alias("month"),
        pl.lit(int(recording.rsplit("mode", 1)[1].split(".", 1)[0]))
        .cast(pl.UInt8)
        .alias("mode"),
    )


def _ppd(experiment: str = "C7-1") -> pl.DataFrame:
    if experiment not in PPD_FILES:
        raise ValueError(
            "experiment must be one of: {}".format(", ".join(sorted(PPD_FILES)))
        )
    return pl.read_csv(
        find_raw_file("ppd", "{}.csv".format(experiment)), try_parse_dates=True
    ).with_columns(pl.lit(experiment).alias("experiment"))


def _ufd(meter: str = "A") -> pl.DataFrame:
    meter = meter.upper()
    if meter not in {"A", "B", "C", "D"}:
        raise ValueError("meter must be A, B, C, or D")
    return read_whitespace(find_raw_file("ufd", "Meter{}.txt".format(meter)))


_LOADERS: dict[str, Loader] = {
    "alpi": _alpi,
    "cbm": _cbm,
    "cmapss": _cmapss,
    "gdd": _gdd,
    "gfd": _gfd,
    "hydsys": _hydsys,
    "ims": _ims,
    "mapm": _mapm,
    "oyicd": _oyicd,
    "ppd": _ppd,
    "ufd": _ufd,
}
