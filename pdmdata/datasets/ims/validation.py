"""Validate every local IMS snapshot without retaining the full dataset."""

from pathlib import Path
import polars as pl
from .loader import inventory, _read_recording

EXPECTED_RECORDINGS = {1: 2156, 2: 984, 3: 6324}


def verify() -> pl.DataFrame:
    """Check channel counts, sample counts, and finite values in every matrix."""
    listing = inventory()
    results = []
    for experiment in (1, 2, 3):
        selected = listing.filter(pl.col("experiment") == experiment)
        if selected.height != EXPECTED_RECORDINGS[experiment]:
            raise ValueError(f"IMS experiment {experiment}: expected "
                             f"{EXPECTED_RECORDINGS[experiment]} recordings, found {selected.height}")
        samples = measurements = 0
        for row in selected.iter_rows(named=True):
            frame = _read_recording(Path(row["path"]), experiment)
            samples += frame.height
            measurements += frame.height * frame.width
        results.append({"experiment": experiment, "recordings": selected.height,
                        "samples": samples, "measurements": measurements,
                        "first_recording": selected["recording_time"][0],
                        "last_recording": selected["recording_time"][-1],
                        "bytes": selected["bytes"].sum()})
    return pl.DataFrame(results)
