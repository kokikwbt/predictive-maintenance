"""Generate an unexecuted MAPM telemetry exploration notebook."""

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
        """# MAPM: Azure predictive-maintenance tables

Load the five Microsoft sample tables, inspect telemetry for one machine,
and plot registered Plotly views.

[Provider](https://github.com/microsoft/sqlworkshops)
· [Dataset guide](../../pdmdata/datasets/mapm/README.md)

Download explicitly first with `pdmdata.download("mapm")` if local files are
missing. Treat the data as synthetic.
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

telemetry = pdmdata.load("mapm", table="telemetry")
errors = pdmdata.load("mapm", table="errors")
failures = pdmdata.load("mapm", table="failures")
machines = pdmdata.load("mapm", table="machines")
maintenance = pdmdata.load("mapm", table="maintenance")
print({name: frame.height for name, frame in {
    "telemetry": telemetry, "errors": errors, "failures": failures,
    "machines": machines, "maintenance": maintenance,
}.items()})
display(machines.head())
display(telemetry.head())
""",
    )
    add(
        "markdown",
        """## One machine trajectory

Filter telemetry and errors for a single `machineID`, then reuse the shared
Plotly registry.
""",
    )
    add(
        "code",
        """MACHINE = 1
unit = telemetry.filter(telemetry["machineID"] == MACHINE)
unit_errors = errors.filter(errors["machineID"] == MACHINE)
print(f"Telemetry rows for machine {MACHINE}: {unit.height:,}")
print(f"Error events: {unit_errors.height:,}")
figure = pdmdata.visualize("mapm", "telemetry", unit, entity=MACHINE)
figure.show()
if unit_errors.height:
    events = pdmdata.visualize("mapm", "error_timeline", unit_errors)
    events.show()
""",
    )

    path = ROOT / "notebooks" / "datasets" / "mapm.ipynb"
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
