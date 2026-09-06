"""IMS vibration snapshots with experiment and recording selectors."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

import polars as pl

from ..config import get_settings

SAMPLING_HZ = 20_000
SAMPLES_PER_RECORDING = 20_480
EXPERIMENT_DIRS = {1: "1st_test", 2: "2nd_test", 3: "3rd_test"}
CHANNEL_COUNTS = {1: 8, 2: 4, 3: 4}
TIMESTAMP_FORMAT = "%Y.%m.%d.%H.%M.%S"
_RECORDING_NAME = re.compile(r"\d{4}(?:\.\d{2}){5}")


def _experiment(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in EXPERIMENT_DIRS:
        raise ValueError("experiment must be 1, 2, or 3")
    return value


def inventory(experiment: int | None = None) -> pl.DataFrame:
    """List local snapshots in time order without reading vibration matrices.

    Recording indices are zero-based within each experiment. Filename timestamps
    are timezone-naive source values; they are not assumed to be UTC.
    """
    selected = list(EXPERIMENT_DIRS) if experiment is None else [_experiment(experiment)]
    directory = get_settings().data_root / "ims" / "extracted"
    files = {number: [] for number in selected}
    for path in directory.rglob("*"):
        if any(part.startswith(".pdmdata-extract-") for part in path.relative_to(directory).parts):
            continue
        if not _RECORDING_NAME.fullmatch(path.name) or not path.is_file():
            continue
        for number in selected:
            if EXPERIMENT_DIRS[number] in path.relative_to(directory).parts[:-1]:
                files[number].append(path)
                break
    rows = []
    for number, paths in files.items():
        if not paths:
            raise FileNotFoundError(
                f"No IMS experiment {number} recordings under {directory}. "
                "Run pdmdata.download('ims') explicitly first."
            )
        names = [p.name for p in paths]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate IMS recording names in experiment {number}")
        for index, path in enumerate(sorted(paths, key=lambda p: p.name)):
            rows.append({"experiment": number, "recording_index": index,
                         "recording": path.name,
                         "recording_time": datetime.strptime(path.name, TIMESTAMP_FORMAT),
                         "channels": CHANNEL_COUNTS[number], "bytes": path.stat().st_size,
                         "path": str(path)})
    return pl.DataFrame(rows, schema_overrides={"experiment": pl.UInt8,
                        "recording_index": pl.UInt32, "channels": pl.UInt8})


def channel_info(experiment: int = 2) -> pl.DataFrame:
    """Map source channels to bearings and documented end-of-test faults.

    These fault descriptions are experiment outcomes, not per-sample labels.
    The source does not specify a physical axis name for each channel.
    """
    experiment = _experiment(experiment)
    faults = {1: {3: "inner race", 4: "roller element"},
              2: {1: "outer race"}, 3: {3: "outer race"}}[experiment]
    rows = []
    for channel in range(1, CHANNEL_COUNTS[experiment] + 1):
        bearing = (channel + 1) // 2 if experiment == 1 else channel
        rows.append({"channel": f"channel_{channel}", "bearing": bearing,
                     "end_of_test_fault": faults.get(bearing, "not reported")})
    return pl.DataFrame(rows)


def _read_recording(path: Path, experiment: int) -> pl.DataFrame:
    width = CHANNEL_COUNTS[experiment]
    frame = pl.read_csv(path, separator="\t", has_header=False, infer_schema=False)
    if frame.width == width + 1 and frame.to_series(-1).null_count() == frame.height:
        frame = frame.select(frame.columns[:-1])
    if frame.shape != (SAMPLES_PER_RECORDING, width):
        raise ValueError(f"Unexpected IMS matrix shape in {path}: {frame.shape}; "
                         f"expected {(SAMPLES_PER_RECORDING, width)}")
    frame = frame.cast(pl.Float64)
    if frame.null_count().to_numpy().sum() or not all(frame.select(pl.all().is_finite().all()).row(0)):
        raise ValueError(f"Missing or non-finite IMS measurements in {path}")
    frame.columns = [f"channel_{i}" for i in range(1, width + 1)]
    return frame


def load(recording: int | str = 0, *, experiment: int | None = None,
         channels: list[str] | None = None) -> pl.DataFrame:
    """Load one complete snapshot; default to recording 0 of experiment 2.

    Pass a zero-based index or an exact timestamp filename. A filename without
    an experiment is looked up across all three experiments. No snapshots are
    concatenated and no data is downloaded implicitly.
    """
    if isinstance(recording, bool) or not isinstance(recording, (int, str)):
        raise ValueError("recording must be a zero-based index or timestamp filename")
    if experiment is None and isinstance(recording, int):
        experiment = 2
    listing = inventory(experiment)
    if isinstance(recording, int):
        if not 0 <= recording < listing.height:
            raise ValueError(f"recording index must be between 0 and {listing.height - 1}")
        row = listing.row(recording, named=True)
    else:
        matches = listing.filter(pl.col("recording") == recording)
        if matches.height != 1:
            raise ValueError(f"Expected one matching IMS recording, found {matches.height}: {recording!r}")
        row = matches.row(0, named=True)
    frame = _read_recording(Path(row["path"]), row["experiment"])
    if channels is not None:
        if not channels or len(set(channels)) != len(channels) or set(channels) - set(frame.columns):
            raise ValueError(f"channels must be unique names from {frame.columns}")
        frame = frame.select(channels)
    return frame.with_row_index("sample").with_columns(
        (pl.col("sample") / SAMPLING_HZ).alias("time_s"),
        pl.lit(row["recording_time"]).alias("recording_time"),
        pl.lit(row["experiment"], dtype=pl.UInt8).alias("experiment"),
        pl.lit(row["recording"]).alias("recording"),
    )


def rms_history(experiment: int = 2, *, every: int = 1) -> pl.DataFrame:
    """Compute channel RMS per selected snapshot, keeping filename timestamps.

    Read one matrix at a time. ``every`` selects evenly spaced recording indices
    and always includes the final recording; it does not resample raw signals.
    """
    experiment = _experiment(experiment)
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError("every must be a positive integer")
    listing = inventory(experiment)
    indices = sorted(set(range(0, listing.height, every)) | {listing.height - 1})
    rows = []
    for index in indices:
        row = listing.row(index, named=True)
        frame = _read_recording(Path(row["path"]), experiment)
        rms = frame.select(pl.all().pow(2).mean().sqrt()).row(0, named=True)
        rows.append({"recording_index": index, "recording_time": row["recording_time"], **rms})
    return pl.DataFrame(rows)
