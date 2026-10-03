"""XJTU-SY bearing run-to-failure snapshots with derived minute-level RUL."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import polars as pl

from pdmdata.config import get_settings


SAMPLING_HZ = 25_600
SAMPLES_PER_SNAPSHOT = 32_768
CONDITION_DIRS = ("35Hz12kN", "37.5Hz11kN", "40Hz10kN")
_SNAPSHOT_NAME = re.compile(r"(\d+)\.csv\Z", re.IGNORECASE)

# Operating conditions and end-of-test fault elements from the dataset
# documentation (Wang et al., IEEE Trans. Reliability 2020).
BEARING_INFO: Dict[str, Dict[str, object]] = {
    "Bearing1_1": {
        "condition": "35Hz12kN",
        "rpm": 2100,
        "load_kn": 12.0,
        "fault": "outer race",
    },
    "Bearing1_2": {
        "condition": "35Hz12kN",
        "rpm": 2100,
        "load_kn": 12.0,
        "fault": "outer race",
    },
    "Bearing1_3": {
        "condition": "35Hz12kN",
        "rpm": 2100,
        "load_kn": 12.0,
        "fault": "outer race",
    },
    "Bearing1_4": {
        "condition": "35Hz12kN",
        "rpm": 2100,
        "load_kn": 12.0,
        "fault": "cage",
    },
    "Bearing1_5": {
        "condition": "35Hz12kN",
        "rpm": 2100,
        "load_kn": 12.0,
        "fault": "inner race and outer race",
    },
    "Bearing2_1": {
        "condition": "37.5Hz11kN",
        "rpm": 2250,
        "load_kn": 11.0,
        "fault": "inner race",
    },
    "Bearing2_2": {
        "condition": "37.5Hz11kN",
        "rpm": 2250,
        "load_kn": 11.0,
        "fault": "outer race",
    },
    "Bearing2_3": {
        "condition": "37.5Hz11kN",
        "rpm": 2250,
        "load_kn": 11.0,
        "fault": "cage",
    },
    "Bearing2_4": {
        "condition": "37.5Hz11kN",
        "rpm": 2250,
        "load_kn": 11.0,
        "fault": "outer race",
    },
    "Bearing2_5": {
        "condition": "37.5Hz11kN",
        "rpm": 2250,
        "load_kn": 11.0,
        "fault": "outer race",
    },
    "Bearing3_1": {
        "condition": "40Hz10kN",
        "rpm": 2400,
        "load_kn": 10.0,
        "fault": "outer race",
    },
    "Bearing3_2": {
        "condition": "40Hz10kN",
        "rpm": 2400,
        "load_kn": 10.0,
        "fault": "inner race, ball, cage and outer race",
    },
    "Bearing3_3": {
        "condition": "40Hz10kN",
        "rpm": 2400,
        "load_kn": 10.0,
        "fault": "inner race",
    },
    "Bearing3_4": {
        "condition": "40Hz10kN",
        "rpm": 2400,
        "load_kn": 10.0,
        "fault": "inner race",
    },
    "Bearing3_5": {
        "condition": "40Hz10kN",
        "rpm": 2400,
        "load_kn": 10.0,
        "fault": "outer race",
    },
}
BEARINGS = tuple(BEARING_INFO)


def _root() -> Path:
    return get_settings().data_root / "xjtu_sy"


def _has_bearing_csvs(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    for path in directory.rglob("1.csv"):
        if path.parent.name.startswith("Bearing"):
            return True
    return False


def _extracted_root() -> Path:
    root = _root()
    nested = root / "extracted" / "XJTU-SY_Bearing_Datasets"
    if _has_bearing_csvs(nested):
        return nested
    extracted = root / "extracted"
    if _has_bearing_csvs(extracted):
        return extracted
    # Fall back to the conventional extract directory for error messages.
    return nested if nested.parent.is_dir() else extracted


def _bearing(value: str) -> str:
    if value not in BEARING_INFO:
        raise ValueError(
            "bearing must be one of: {}".format(", ".join(BEARINGS))
        )
    return value


def _condition(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value not in CONDITION_DIRS:
        raise ValueError(
            "condition must be one of: {}".format(", ".join(CONDITION_DIRS))
        )
    return value


def _bearing_dir(bearing: str) -> Path:
    bearing = _bearing(bearing)
    condition = str(BEARING_INFO[bearing]["condition"])
    root = _extracted_root()
    candidates = [
        root / condition / bearing,
        root / bearing,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        "Bearing directory for {!r} was not found under {}. Run "
        "pdmdata.download('xjtu_sy') first.".format(bearing, root)
    )


def _snapshot_paths(bearing: str) -> List[Path]:
    directory = _bearing_dir(bearing)
    paths = []
    for path in directory.iterdir():
        match = _SNAPSHOT_NAME.fullmatch(path.name)
        if match and path.is_file():
            paths.append((int(match.group(1)), path))
    if not paths:
        raise FileNotFoundError(
            "No CSV snapshots found for {!r} under {}".format(
                bearing, directory
            )
        )
    paths.sort(key=lambda item: item[0])
    indices = [index for index, _ in paths]
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError(
            "Snapshot files for {!r} must be consecutive 1.csv..N.csv".format(
                bearing
            )
        )
    return [path for _, path in paths]


def _read_waveform(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    # Official CSVs include a one-line header; synthetic fixtures may omit it.
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline()
    skiprows = 0
    try:
        [float(part) for part in first.strip().split(",")[:2]]
    except ValueError:
        skiprows = 1
    values = np.loadtxt(
        path, delimiter=",", dtype=np.float64, skiprows=skiprows
    )
    if values.ndim == 1:
        raise ValueError(
            "{} must contain two columns (horizontal, vertical)".format(path)
        )
    if values.shape[1] != 2:
        raise ValueError(
            "{} must have exactly two columns; found {}".format(
                path, values.shape[1]
            )
        )
    return values[:, 0], values[:, 1]


def available_bearings() -> Tuple[str, ...]:
    """Return bearing ids present under the local extract tree."""
    root = _extracted_root()
    found = []
    for bearing, info in BEARING_INFO.items():
        condition = str(info["condition"])
        if (root / condition / bearing).is_dir() or (root / bearing).is_dir():
            found.append(bearing)
    return tuple(found)


def inventory(condition: Optional[str] = None) -> pl.DataFrame:
    """List local bearings with snapshot counts and derived lifetimes."""
    condition = _condition(condition)
    rows = []
    for bearing, info in BEARING_INFO.items():
        if condition is not None and info["condition"] != condition:
            continue
        try:
            paths = _snapshot_paths(bearing)
        except FileNotFoundError:
            continue
        rows.append(
            {
                "bearing": bearing,
                "condition": info["condition"],
                "rpm": info["rpm"],
                "load_kn": info["load_kn"],
                "fault": info["fault"],
                "snapshots": len(paths),
                "lifetime_min": len(paths),
                "directory": str(_bearing_dir(bearing)),
            }
        )
    if not rows:
        raise FileNotFoundError(
            "No XJTU-SY bearings found under {}. Run "
            "pdmdata.download('xjtu_sy') first.".format(_extracted_root())
        )
    return pl.DataFrame(rows)


def load(
    bearing: str = "Bearing1_1",
    *,
    cycle: Optional[int] = None,
    with_waveform: bool = False,
    with_rul: bool = True,
) -> pl.DataFrame:
    """Load one bearing as snapshot-level rows, or one raw waveform.

    By default each row is one minute-level CSV snapshot with RMS features and
    derived ``RUL`` (minutes until the final snapshot). Set
    ``with_waveform=True`` and ``cycle`` to return the 32,768-sample waveform
    for that snapshot instead.
    """
    bearing = _bearing(bearing)
    paths = _snapshot_paths(bearing)
    info = BEARING_INFO[bearing]
    n_snapshots = len(paths)

    if with_waveform:
        if cycle is None:
            raise ValueError("cycle is required when with_waveform=True")
        if (
            isinstance(cycle, bool)
            or not isinstance(cycle, int)
            or cycle < 1
            or cycle > n_snapshots
        ):
            raise ValueError(
                "cycle must be an integer in 1..{}".format(n_snapshots)
            )
        horizontal, vertical = _read_waveform(paths[cycle - 1])
        frame = pl.DataFrame(
            {
                "bearing": [bearing] * len(horizontal),
                "condition": [info["condition"]] * len(horizontal),
                "cycle": [cycle] * len(horizontal),
                "sample": np.arange(len(horizontal), dtype=np.uint32),
                "time_s": np.arange(len(horizontal), dtype=np.float64)
                / SAMPLING_HZ,
                "horizontal": horizontal,
                "vertical": vertical,
            }
        )
        if with_rul:
            frame = frame.with_columns(
                pl.lit(n_snapshots - cycle).alias("RUL")
            )
        return frame

    if cycle is not None:
        if (
            isinstance(cycle, bool)
            or not isinstance(cycle, int)
            or cycle < 1
            or cycle > n_snapshots
        ):
            raise ValueError(
                "cycle must be an integer in 1..{}".format(n_snapshots)
            )
        selected = [(cycle, paths[cycle - 1])]
    else:
        selected = list(enumerate(paths, start=1))

    rows = []
    for index, path in selected:
        horizontal, vertical = _read_waveform(path)
        row = {
            "bearing": bearing,
            "condition": info["condition"],
            "rpm": info["rpm"],
            "load_kn": info["load_kn"],
            "fault": info["fault"],
            "cycle": index,
            "rms_horizontal": float(np.sqrt(np.mean(np.square(horizontal)))),
            "rms_vertical": float(np.sqrt(np.mean(np.square(vertical)))),
            "n_samples": int(horizontal.shape[0]),
            "path": str(path),
        }
        if with_rul:
            row["RUL"] = n_snapshots - index
        rows.append(row)
    return pl.DataFrame(rows)


def load_many(
    bearings: Optional[Iterable[str]] = None,
    *,
    condition: Optional[str] = None,
    with_rul: bool = True,
) -> pl.DataFrame:
    """Concatenate snapshot-level tables for several bearings."""
    condition = _condition(condition)
    if bearings is None:
        selected = [
            name
            for name, info in BEARING_INFO.items()
            if condition is None or info["condition"] == condition
        ]
        selected = [name for name in selected if name in available_bearings()]
    else:
        selected = [_bearing(name) for name in bearings]
    if not selected:
        raise FileNotFoundError(
            "No matching XJTU-SY bearings are available locally."
        )
    frames = [
        load(name, with_waveform=False, with_rul=with_rul)
        for name in selected
    ]
    return pl.concat(frames, how="vertical_relaxed")
