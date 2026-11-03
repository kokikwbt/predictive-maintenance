#!/usr/bin/env python3
"""Generate the Backblaze drive-history exploration notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notebook_cells import ROOT, NotebookBuilder  # noqa: E402


def main() -> None:
    nb = NotebookBuilder()

    nb.markdown(
        """# Backblaze drive histories

Explore one downloaded quarter using local file inventory, drive selection, and
Matplotlib SMART waveforms. A full quarter requires several GB of disk space.
Download explicitly in a terminal first:

```bash
uv run --locked python scripts/download.py backblaze --variant 2025-q1
```

This notebook does not initiate downloads. Configure data_root in pdmdata.toml.
"""
    )
    nb.code(
        """from pathlib import Path
import sys

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))
import matplotlib.pyplot as plt
import polars as pl
import pdmdata
from pdmdata.datasets.backblaze import inventory, verify
from pdmdata.datasets.backblaze.viz import plot_waveforms

VARIANT = "2025-q1"
inventory(VARIANT).select("date", "bytes")
"""
    )
    nb.markdown(
        "Choose a drive from a single day, then load its history. "
        "Replace SERIAL with a known drive serial number to reproduce a specific experiment."
    )
    nb.code(
        """files = inventory(VARIANT)
first_day = files["date"].min()
candidates = pdmdata.load("backblaze", variant=VARIANT, start_date=first_day,
                          end_date=first_day, columns=["serial_number", "model"])
SERIAL = candidates.head(1).collect()["serial_number"][0]
history = pdmdata.load("backblaze", variant=VARIANT, serial_number=SERIAL)
history.select("date", "serial_number", "model", "failure").collect()
"""
    )
    nb.code(
        """preferred = ["smart_5_raw", "smart_9_raw", "smart_194_raw"]
available = history.collect_schema().names()
columns = [c for c in preferred if c in available]
figure = plot_waveforms(history, serial_number=SERIAL, columns=columns)
plt.show()
"""
    )
    nb.markdown(
        """## Optional verification

`verify(VARIANT, check_archive=True)` reads the complete quarter and checks
calendar coverage, record validity, and extracted-file integrity against the
local ZIP. It may take time. No network access is used.

SMART availability depends on the drive model. Missing values are not zero.
Reported failure dates are marked in red. Disappearance alone is not a failure
label; define censoring explicitly for survival experiments. Saved Matplotlib
outputs can be displayed on GitHub after running and saving this notebook.
"""
    )

    nb.write(ROOT / "notebooks/datasets/backblaze.ipynb")


if __name__ == "__main__":
    main()
