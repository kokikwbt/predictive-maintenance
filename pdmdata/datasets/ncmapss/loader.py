"""N-CMAPSS HDF5 trajectories with per-cycle RUL labels."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import polars as pl

from pdmdata.config import get_settings


VARIANT_FILES = {
    "ds01": "N-CMAPSS_DS01-005.h5",
    "ds02": "N-CMAPSS_DS02-006.h5",
    "ds03": "N-CMAPSS_DS03-012.h5",
    "ds04": "N-CMAPSS_DS04.h5",
    "ds05": "N-CMAPSS_DS05.h5",
    "ds06": "N-CMAPSS_DS06.h5",
    "ds07": "N-CMAPSS_DS07.h5",
    "ds08a": "N-CMAPSS_DS08a-009.h5",
    "ds08c": "N-CMAPSS_DS08c-008.h5",
    "ds08d": "N-CMAPSS_DS08d-010.h5",
}
VARIANTS = tuple(VARIANT_FILES)
SPLITS = ("dev", "test")


def _variant(value: str) -> str:
    if value not in VARIANT_FILES:
        raise ValueError(
            "variant must be one of: {}".format(", ".join(VARIANTS))
        )
    return value


def _split(value: str) -> str:
    if value == "train":
        return "dev"
    if value not in SPLITS:
        raise ValueError("split must be dev, test, or train (alias of dev)")
    return value


def _path(variant: str) -> Path:
    root = get_settings().data_root / "ncmapss" / variant
    filename = VARIANT_FILES[variant]
    candidates = [
        root / filename,
        root / "extracted" / filename,
        root / "data_set" / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "{} was not found under {}. Run "
        "pdmdata.download('ncmapss', variant={!r}) first.".format(
            filename, root, variant
        )
    )


def _decode_names(values: np.ndarray) -> list[str]:
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def load(
    variant: str = "ds01",
    split: str = "dev",
    *,
    unit: int | None = None,
    include_virtual: bool = False,
    include_health_modifiers: bool = False,
    with_rul: bool = True,
) -> pl.DataFrame:
    """Load one N-CMAPSS variant/split as a Polars DataFrame.

    The HDF5 ``dev`` split is the development/training fleet; ``train`` is
    accepted as an alias. Per-cycle RUL is stored in ``Y_*`` and exposed as
    ``RUL`` when ``with_rul=True``. Virtual sensors (``X_v``) and health
    modifiers (``T``) are omitted unless requested because they are not always
    used as model inputs.
    """
    variant = _variant(variant)
    split = _split(split)
    if unit is not None and (
        isinstance(unit, bool) or not isinstance(unit, int) or unit < 1
    ):
        raise ValueError("unit must be a positive integer")

    path = _path(variant)
    with h5py.File(path, "r") as handle:
        aux = np.asarray(handle[f"A_{split}"])
        flight = np.asarray(handle[f"W_{split}"])
        sensors = np.asarray(handle[f"X_s_{split}"])
        rul = np.asarray(handle[f"Y_{split}"]).reshape(-1)
        aux_names = _decode_names(handle["A_var"][()])
        flight_names = _decode_names(handle["W_var"][()])
        sensor_names = _decode_names(handle["X_s_var"][()])
        virtual = None
        virtual_names: list[str] = []
        modifiers = None
        modifier_names: list[str] = []
        if include_virtual:
            virtual = np.asarray(handle[f"X_v_{split}"])
            virtual_names = _decode_names(handle["X_v_var"][()])
        if include_health_modifiers:
            modifiers = np.asarray(handle[f"T_{split}"])
            modifier_names = _decode_names(handle["T_var"][()])

    if unit is not None:
        unit_column = aux_names.index("unit")
        mask = aux[:, unit_column] == unit
        if not np.any(mask):
            raise ValueError(
                f"No unit {unit} in {variant} {split}"
            )
        aux = aux[mask]
        flight = flight[mask]
        sensors = sensors[mask]
        rul = rul[mask]
        if virtual is not None:
            virtual = virtual[mask]
        if modifiers is not None:
            modifiers = modifiers[mask]

    data: dict[str, np.ndarray] = {}
    for index, name in enumerate(aux_names):
        column = "unit_number" if name == "unit" else name
        data[column] = aux[:, index]
    for index, name in enumerate(flight_names):
        data[name] = flight[:, index]
    for index, name in enumerate(sensor_names):
        data[name] = sensors[:, index]
    if virtual is not None:
        for index, name in enumerate(virtual_names):
            data[name] = virtual[:, index]
    if modifiers is not None:
        for index, name in enumerate(modifier_names):
            data[name] = modifiers[:, index]
    if with_rul:
        data["RUL"] = rul

    frame = pl.DataFrame(data)
    return frame.with_columns(
        pl.col("unit_number").cast(pl.UInt16),
        pl.col("cycle").cast(pl.UInt32),
        *(
            [pl.col("RUL").cast(pl.Int64)]
            if with_rul
            else []
        ),
    )


def inventory(variant: str | None = None) -> pl.DataFrame:
    """Summarize units, rows, and RUL ranges for local N-CMAPSS files."""
    selected = VARIANTS if variant is None else (_variant(variant),)
    rows = []
    for name in selected:
        try:
            path = _path(name)
        except FileNotFoundError:
            continue
        with h5py.File(path, "r") as handle:
            for split in SPLITS:
                aux = np.asarray(handle[f"A_{split}"])
                rul = np.asarray(handle[f"Y_{split}"]).reshape(-1)
                units = np.unique(aux[:, 0])
                rows.append(
                    {
                        "variant": name,
                        "split": split,
                        "units": int(units.size),
                        "rows": int(aux.shape[0]),
                        "min_rul": int(rul.min()),
                        "max_rul": int(rul.max()),
                        "file": path.name,
                    }
                )
    if not rows:
        raise FileNotFoundError(
            "No local N-CMAPSS variants found. Download one with "
            "pdmdata.download('ncmapss', variant='ds01')."
        )
    return pl.DataFrame(rows)


def available_variants() -> list[str]:
    """Return locally downloaded N-CMAPSS variant IDs."""
    found = []
    for variant in VARIANTS:
        try:
            _path(variant)
        except FileNotFoundError:
            continue
        found.append(variant)
    return found
