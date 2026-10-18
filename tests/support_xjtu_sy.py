"""Shared XJTU-SY fixtures for loader and task tests."""

from pathlib import Path

import numpy as np

from pdmdata.datasets.xjtu_sy import loader as xjtu_loader


def write_bearing(
    root: Path,
    bearing: str,
    n_snapshots: int,
    *,
    samples: int = 8,
) -> None:
    info = xjtu_loader.BEARING_INFO[bearing]
    directory = (
        root / "xjtu_sy" / "extracted" / str(info["condition"]) / bearing
    )
    directory.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 1, samples, endpoint=False)
    for cycle in range(1, n_snapshots + 1):
        horizontal = (0.1 * cycle) * np.sin(2 * np.pi * t)
        vertical = (0.05 * cycle) * np.cos(2 * np.pi * t)
        path = directory / "{}.csv".format(cycle)
        np.savetxt(
            path,
            np.column_stack([horizontal, vertical]),
            delimiter=",",
        )
