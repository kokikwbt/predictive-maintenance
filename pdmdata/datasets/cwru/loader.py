"""Source-faithful CWRU discovery and loading across all four collections."""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError

from pdmdata.config import get_settings

COLLECTIONS = ("normal", "12k_de", "48k_de", "12k_fe")
CHANNELS = ("DE", "FE", "BA")


def _manifest() -> pl.DataFrame:
    with Path(__file__).with_name("recordings.json").open() as stream:
        return pl.DataFrame(json.load(stream))


def _values(value, name):
    values = list(value) if isinstance(value, (list, tuple, set)) else [value]
    if not values or any(isinstance(item, bool) for item in values):
        raise ValueError(f"{name} must contain at least one valid value")
    return values


def _local_paths() -> dict[int, list[Path]]:
    root = get_settings().data_root / "cwru"
    paths: dict[int, list[Path]] = {}
    for path in sorted(root.rglob("*.mat")):
        if not path.stem.isdigit() or not path.is_file():
            continue
        parts = path.relative_to(root).parts[:-1]
        if any(part.startswith(("tmp", ".")) for part in parts):
            continue
        paths.setdefault(int(path.stem), []).append(path)
    return paths


def inventory(
    *,
    recording: int | Sequence[int] | None = None,
    collection: str | Sequence[str] | None = None,
    load_hp: int | Sequence[int] | None = None,
    fault: str | Sequence[str] | None = None,
    diameter_in: float | Sequence[float] | None = None,
    outer_race_position: int | Sequence[int] | None = None,
    installed: bool | None = None,
) -> pl.DataFrame:
    """List official recordings with AND filters and OR within each list.

    By default, inventory needs no local files. installed=True/False adds a
    filesystem scan and selects present/missing records. No data is downloaded.
    """
    frame = _manifest()
    mask = pl.lit(True)
    for column, value in (
        ("recording", recording),
        ("collection", collection),
        ("load_hp", load_hp),
        ("fault", fault),
        ("diameter_in", diameter_in),
        ("outer_race_position", outer_race_position),
    ):
        if value is None:
            continue
        values = _values(value, column)
        allowed = frame[column].drop_nulls().to_list()
        if any(item not in allowed for item in values):
            raise ValueError(f"Invalid {column}: {value!r}")
        if column in {"recording", "load_hp", "outer_race_position"} and any(
            type(item) is not int for item in values
        ):
            raise ValueError(f"{column} requires integer values")
        mask = mask & pl.col(column).is_in(values)
    frame = frame.filter(mask).sort("recording")
    if installed is not None:
        if type(installed) is not bool:
            raise ValueError("installed must be True, False, or None")
        frame = frame.filter(
            pl.col("recording").is_in(list(_local_paths())) == installed
        )
    return frame


def _path(info: dict) -> Path:
    paths = _local_paths().get(info["recording"], [])
    if not paths:
        raise FileNotFoundError(
            f"CWRU {info['filename']} is missing. Run "
            f"cwru.download(recording={info['recording']}) explicitly first."
        )
    if len(paths) > 1:
        digests = {hashlib.sha256(path.read_bytes()).digest() for path in paths}
        if len(digests) != 1:
            raise ValueError(
                f"Conflicting local copies of {info['filename']}: {paths}"
            )
    return paths[0]


def _info(recording: int) -> dict:
    if type(recording) is not int:
        raise ValueError("recording must be an integer from cwru.inventory()")
    return inventory(recording=recording).row(0, named=True)


def _read(info: dict):
    path = _path(info)
    try:
        source = loadmat(path)
    except (OSError, ValueError, IndexError, EOFError, MatReadError) as error:
        raise ValueError(
            f"Invalid MATLAB source {path.name}: {error}"
        ) from error
    signals = {}
    for channel in CHANNELS:
        keys = [key for key in source if key.endswith(f"_{channel}_time")]
        preferred = f"X{info['recording']:03d}_{channel}_time"
        if preferred in keys:
            keys = [preferred]
        if len(keys) > 1:
            raise ValueError(
                f"Multiple {channel} variables in {path.name}: {keys}"
            )
        if not keys:
            continue
        key = keys[0]
        signal = np.asarray(source[key])
        if signal.ndim != 2 or 1 not in signal.shape or signal.size == 0:
            raise ValueError(
                f"{key} must be a nonempty MATLAB row or column vector"
            )
        if signal.dtype.kind not in "fiu" or not np.isfinite(signal).all():
            raise ValueError(f"{key} must contain finite real numeric samples")
        signals[channel] = (key, signal.reshape(-1))
    if not signals:
        raise ValueError(
            f"No documented vibration channels found in {path.name}"
        )
    keys = [key for key in source if key.endswith("RPM")]
    preferred = f"X{info['recording']:03d}RPM"
    if preferred in keys:
        keys = [preferred]
    if len(keys) > 1:
        raise ValueError("Multiple RPM variables found in the recording")
    rpm = None
    if keys:
        value = np.asarray(source[keys[0]])
        if (
            value.size != 1
            or value.dtype.kind not in "fiu"
            or not np.isfinite(value).all()
        ):
            raise ValueError("RPM must be a finite numeric scalar")
        rpm = float(value.item())
    return path, signals, rpm


def variables(recording: int = 105) -> pl.DataFrame:
    """List every MATLAB variable, including auxiliary and embedded series."""
    path = _path(_info(recording))
    return pl.DataFrame(
        [
            {"variable": name, "shape": list(shape), "matlab_class": kind}
            for name, shape, kind in whosmat(path)
        ]
    )


def load_variable(recording: int, variable: str) -> np.ndarray:
    """Return one raw MATLAB array without flattening, casting or relabeling."""
    if (
        not isinstance(variable, str)
        or variable not in variables(recording)["variable"]
    ):
        raise ValueError(f"Unknown MATLAB variable: {variable!r}")
    return loadmat(_path(_info(recording)), variable_names=[variable])[variable]


def _rate(info: dict, channel: str, sample_rates: Mapping | None):
    if sample_rates is not None:
        if not isinstance(sample_rates, Mapping):
            raise ValueError(
                "sample_rates must map DE, FE, or BA to positive rates"
            )
        for key, value in sample_rates.items():
            if key not in CHANNELS or type(value) is not int or value <= 0:
                raise ValueError(
                    "sample_rates must map DE, FE, or BA to positive integer rates"
                )
        if channel in sample_rates:
            return sample_rates[channel], "user_override"
    # Collection headings do not establish every channel's physical rate.
    # The 0.028 inch files on the 12k page have ambiguous acquisition rates.
    if info["collection"] == "12k_de" and info["diameter_in"] == 0.028:
        return None, "unspecified"
    if info["collection"] in {"12k_de", "12k_fe"}:
        return 12000, "collection_documentation"
    if info["collection"] == "48k_de" and channel == "DE":
        return 48000, "collection_documentation"
    return None, "unspecified"


def channel_info(
    recording: int = 105, *, sample_rates: Mapping | None = None
) -> pl.DataFrame:
    """Inspect installed source channels, lengths and documented sample rates."""
    info = _info(recording)
    path, signals, rpm = _read(info)
    rows = []
    for channel, (variable, signal) in signals.items():
        rate, origin = _rate(info, channel, sample_rates)
        rows.append(
            {
                "recording": recording,
                "channel": channel,
                "source_variable": variable,
                "samples": signal.size,
                "sample_rate_hz": rate,
                "rate_source": origin,
                "rpm": rpm,
                "path": str(path),
            }
        )
    return pl.DataFrame(
        rows, schema_overrides={"sample_rate_hz": pl.UInt32, "rpm": pl.Float64}
    )


def load(
    recording: int = 105,
    *,
    channels: str | Sequence[str] = "DE",
    start: int = 0,
    stop: int | None = None,
    step: int = 1,
    sample_rates: Mapping[str, int] | None = None,
) -> pl.DataFrame:
    """Load one recording in long form with independent channel sample indices.

    channels='all' preserves unequal channel lengths without padding/resampling.
    start/stop/step slice original sample indices, with exclusive stop. MATLAB
    parsing still reads the whole file. Unspecified rates give null time_s.
    """
    for name, value in (("start", start), ("step", step), ("stop", stop)):
        if name == "stop" and value is None:
            continue
        if type(value) is not int or value < (1 if name == "step" else 0):
            raise ValueError(f"Invalid {name}: {value!r}")
    if stop is not None and stop <= start:
        raise ValueError("stop must be greater than start")
    selected = None if channels == "all" else _values(channels, "channels")
    if selected is not None and (
        len(set(selected)) != len(selected)
        or any(c not in CHANNELS for c in selected)
    ):
        raise ValueError(
            "channels must be DE, FE, BA, 'all', or a unique list of channels"
        )
    info = _info(recording)
    _, signals, rpm = _read(info)
    selected = list(signals) if selected is None else selected
    missing = set(selected) - signals.keys()
    if missing:
        raise ValueError(
            f"Missing channels {sorted(missing)}; available: {list(signals)}"
        )
    frames = []
    for channel in selected:
        variable, values = signals[channel]
        indices = np.arange(
            start,
            min(stop if stop is not None else values.size, values.size),
            step,
        )
        if not indices.size:
            raise ValueError(f"No {channel} samples in the requested slice")
        rate, origin = _rate(info, channel, sample_rates)
        frames.append(
            pl.DataFrame(
                {
                    "sample": pl.Series(indices, dtype=pl.UInt32),
                    "amplitude": values[indices].astype(np.float64),
                }
            ).with_columns(
                pl.lit(recording, dtype=pl.UInt32).alias("recording"),
                pl.lit(info["load_hp"], dtype=pl.UInt8).alias("load_hp"),
                pl.lit(info["fault"]).alias("fault"),
                pl.lit(info["diameter_in"]).alias("diameter_in"),
                pl.lit(info["outer_race_position"], dtype=pl.Int64).alias(
                    "outer_race_position"
                ),
                pl.lit(info["collection"]).alias("collection"),
                pl.lit(info["fault_location"], dtype=pl.String).alias(
                    "fault_location"
                ),
                pl.lit(channel).alias("channel"),
                pl.lit(rate, dtype=pl.UInt32).alias("sample_rate_hz"),
                pl.lit(origin).alias("rate_source"),
                pl.lit(rpm, dtype=pl.Float64).alias("rpm"),
                pl.lit(variable).alias("source_variable"),
                (
                    pl.col("sample") / rate
                    if rate
                    else pl.lit(None, dtype=pl.Float64)
                ).alias("time_s"),
            )
        )
    return pl.concat(frames)


def iter_load(
    *,
    recording=None,
    collection=None,
    load_hp=None,
    fault=None,
    diameter_in=None,
    outer_race_position=None,
    channels="DE",
    start=0,
    stop=None,
    step=1,
    sample_rates=None,
) -> Iterator[pl.DataFrame]:
    """Yield one selected recording at a time; never join different trajectories."""
    selected = inventory(
        recording=recording,
        collection=collection,
        load_hp=load_hp,
        fault=fault,
        diameter_in=diameter_in,
        outer_race_position=outer_race_position,
    )
    if selected.is_empty():
        raise ValueError("No CWRU recordings match the selection")
    for identifier in selected["recording"]:
        yield load(
            identifier,
            channels=channels,
            start=start,
            stop=stop,
            step=step,
            sample_rates=sample_rates,
        )
