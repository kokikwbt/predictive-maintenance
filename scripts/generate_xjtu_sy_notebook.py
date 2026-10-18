"""Generate an unexecuted XJTU-SY bearing exploration notebook."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    cells = []

    def add(kind, source):
        cell = dict(
            cell_type=kind,
            id=f"cell-{len(cells):02d}",
            metadata={},
            source=source.splitlines(True),
        )
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        """# XJTU-SY: bearing run-to-failure snapshots

List local bearings, load minute-level RMS features with derived RUL, and plot
the shared Plotly RMS trajectory.

[Dataset guide](../../pdmdata/datasets/xjtu_sy/README.md)

Download explicitly first with `pdmdata.download("xjtu_sy")`. The archive is
large and excluded from bulk download.
""",
    )
    add(
        "code",
        """from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))
from IPython.display import display
import pdmdata
from pdmdata.datasets.xjtu_sy import available_bearings, inventory

print("Bearings:", available_bearings()[:5], "...")
display(inventory())
BEARING = available_bearings()[0]
history = pdmdata.load("xjtu_sy", bearing=BEARING, with_rul=True)
display(history.select(
    ["bearing", "cycle", "rms_horizontal", "rms_vertical", "RUL"]
).head())
figure = pdmdata.visualize("xjtu_sy", "rms_trajectory", history)
figure.show()
""",
    )

    path = ROOT / "notebooks" / "datasets" / "xjtu_sy.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {
                    "kernelspec": {
                        "display_name": "Python (pdmdata)",
                        "language": "python",
                        "name": "pdmdata",
                    },
                    "language_info": {
                        "name": "python",
                        "pygments_lexer": "ipython3",
                    },
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    print(path)


if __name__ == "__main__":
    main()
