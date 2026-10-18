"""Generate an unexecuted N-CMAPSS exploration notebook."""

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
        """# N-CMAPSS: run-to-failure trajectories

Inventory a local variant, load one unit with per-cycle RUL, and plot sensors
through the shared Plotly API.

[Dataset guide](../../pdmdata/datasets/ncmapss/README.md)

Download a variant explicitly first, for example
`pdmdata.download("ncmapss", variant="ds01")`. Large files are opt-in and
excluded from bulk download.
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
from pdmdata.datasets.ncmapss import available_variants, inventory

print("Variants:", available_variants())
summary = inventory()
display(summary)
VARIANT = "ds01"
frame = pdmdata.load("ncmapss", variant=VARIANT, split="dev", with_rul=True)
units = frame["unit_number"].unique().sort().to_list()
print(f"{VARIANT} dev units: {len(units)}; rows: {frame.height:,}")
UNIT = units[0]
unit = frame.filter(frame["unit_number"] == UNIT)
display(unit.select(["unit_number", "cycle", "RUL"]).head())
figure = pdmdata.visualize(
    "ncmapss", "sensor_trajectory", unit, entity=UNIT
)
figure.show()
""",
    )

    path = ROOT / "notebooks" / "datasets" / "ncmapss.ipynb"
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
                    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
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
