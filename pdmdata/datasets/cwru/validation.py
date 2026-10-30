"""Check local CWRU channels and report coverage without downloading."""

import hashlib

import numpy as np
import polars as pl

from .loader import _read, inventory, load


def validate(
    *,
    recording=None,
    collection=None,
    load_hp=None,
    fault=None,
    diameter_in=None,
    outer_race_position=None,
    installed_only: bool = False,
) -> pl.DataFrame:
    """Validate selected files and exact adapter/source equality for each channel.

    Fail on missing files unless installed_only=True. Source and adapter use
    the same SciPy reader, so this is not independent-parser validation.
    Physical acquisition rates and experiment labels are not inferred here.
    """
    if type(installed_only) is not bool:
        raise ValueError("installed_only must be True or False")
    selected = inventory(
        recording=recording,
        collection=collection,
        load_hp=load_hp,
        fault=fault,
        diameter_in=diameter_in,
        outer_race_position=outer_race_position,
        installed=True if installed_only else None,
    )
    if selected.is_empty():
        raise ValueError("No CWRU recordings match the validation selection")
    rows = []
    for info in selected.iter_rows(named=True):
        path, signals, rpm = _read(info)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = load(info["recording"], channels="all")
        for channel, (variable, values) in signals.items():
            observed = frame.filter(pl.col("channel") == channel)
            if not np.array_equal(observed["amplitude"].to_numpy(), values):
                raise ValueError(
                    f"Source/adapter mismatch: {path.name}/{channel}"
                )
            rows.append(
                {
                    "recording": info["recording"],
                    "collection": info["collection"],
                    "channel": channel,
                    "source_variable": variable,
                    "samples": len(values),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                    "rpm": rpm,
                    "sample_rate_hz": observed["sample_rate_hz"][0],
                    "exact_array_match": True,
                }
            )
    return pl.DataFrame(
        rows, schema_overrides={"rpm": pl.Float64, "sample_rate_hz": pl.UInt32}
    )
