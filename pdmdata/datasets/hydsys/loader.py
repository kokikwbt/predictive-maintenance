"""Hydraulic-cycle loading with native sampling rates and cycle-level labels."""

from __future__ import annotations

import numpy as np
import polars as pl
from pdmdata.io import find_raw_file

CYCLE_COUNT = 2205
CYCLE_SECONDS = 60
SENSOR_INFO = {
    **{f"PS{i}": (100, "bar") for i in range(1, 7)},
    "EPS1": (100, "W"),
    **{f"FS{i}": (10, "L/min") for i in range(1, 3)},
    **{f"TS{i}": (1, "deg C") for i in range(1, 5)},
    "VS1": (1, "mm/s"), "CE": (1, "%"), "CP": (1, "kW"), "SE": (1, "%"),
}
TARGET_VALUES = {
    "cooler_condition": {3, 20, 100},
    "valve_condition": {73, 80, 90, 100},
    "internal_pump_leakage": {0, 1, 2},
    "hydraulic_accumulator": {90, 100, 115, 130},
    "stable_flag": {0, 1},
}


def _sensor(sensor):
    sensor = sensor.removesuffix(".txt")
    if sensor not in SENSOR_INFO:
        raise ValueError(f"sensor must be one of {tuple(SENSOR_INFO)}")
    return sensor


def _cycle(cycle):
    if type(cycle) is not int or not 0 <= cycle < CYCLE_COUNT:
        raise ValueError(f"cycle must be a zero-based integer from 0 to {CYCLE_COUNT - 1}")
    return cycle


def _matrix(sensor, cycle=None):
    width = SENSOR_INFO[sensor][0] * CYCLE_SECONDS
    frame = pl.read_csv(
        find_raw_file("hydsys", f"{sensor}.txt"), separator="\t", has_header=False,
        schema={f"sample_{i}": pl.Float64 for i in range(width)},
        skip_rows=0 if cycle is None else cycle, n_rows=None if cycle is None else 1,
    )
    if frame.width != width or frame.is_empty():
        raise ValueError(f"Expected {width} measurements per {sensor} cycle")
    if frame.null_count().sum_horizontal().item() or not all(
        frame.select(pl.all().is_finite().all()).row(0)
    ):
        raise ValueError(f"Missing or non-finite {sensor} measurements")
    return frame


def load(sensor: str = "PS1", *, cycle: int | None = None) -> pl.DataFrame:
    """Load a cycle-by-sample matrix, or a single cycle with a native time axis.

    Without cycle, rows represent cycles and columns represent samples. With
    cycle, return sample, time_s, and the sensor value. time_s is nominal elapsed
    time derived from the provider's sampling rate, not a measured timestamp.
    """
    sensor = _sensor(sensor)
    if cycle is not None:
        _cycle(cycle)
    frame = _matrix(sensor, cycle)
    if cycle is None:
        return frame
    rate = SENSOR_INFO[sensor][0]
    values = np.asarray(frame.row(0), dtype=float)
    return pl.DataFrame({"sample": np.arange(len(values), dtype=np.uint32),
                         "time_s": np.arange(len(values)) / rate, sensor: values})


def load_cycle(cycle: int = 0, *, sensors=None) -> dict[int, pl.DataFrame]:
    """Return one frame per native sampling rate, with no resampling.

    The default includes all 17 signals. Labels are accessed separately through
    profile(), indexed by the same zero-based cycle ID.
    """
    _cycle(cycle)
    sensors = list(SENSOR_INFO) if sensors is None else list(sensors)
    if not sensors:
        raise ValueError("Select at least one sensor")
    sensors = [_sensor(sensor) for sensor in sensors]
    if len(set(sensors)) != len(sensors):
        raise ValueError("Select each sensor only once")
    groups = {}
    for sensor in sensors:
        frame = load(sensor, cycle=cycle)
        rate = SENSOR_INFO[sensor][0]
        if rate in groups:
            groups[rate] = groups[rate].with_columns(frame[sensor])
        else:
            groups[rate] = frame
    return groups


def profile() -> pl.DataFrame:
    """Load four condition targets and the stability flag, one row per cycle."""
    frame = pl.read_csv(find_raw_file("hydsys", "profile.txt"), separator="\t",
                        has_header=False, schema={name: pl.Int16 for name in TARGET_VALUES})
    if frame.shape != (CYCLE_COUNT, 5) or frame.null_count().sum_horizontal().item():
        raise ValueError(f"Expected {CYCLE_COUNT} complete HYDSYS label rows")
    for column, allowed in TARGET_VALUES.items():
        if not set(frame[column].unique().to_list()) <= allowed:
            raise ValueError(f"Unexpected HYDSYS target values in {column}")
    return frame.with_row_index("cycle")


def inventory() -> pl.DataFrame:
    """Validate every sensor matrix against the profile and report native shapes."""
    labels = profile()
    rows = []
    for sensor, (rate, unit) in SENSOR_INFO.items():
        frame = load(sensor)
        if frame.height != labels.height:
            raise ValueError(f"{sensor} cycle count does not match profile.txt")
        path = find_raw_file("hydsys", f"{sensor}.txt")
        rows.append({"sensor": sensor, "sampling_hz": rate, "unit": unit,
                     "cycles": frame.height, "samples_per_cycle": frame.width,
                     "measurements": frame.height * frame.width, "bytes": path.stat().st_size})
    return pl.DataFrame(rows)
