"""Generate an unexecuted MAPM telemetry exploration notebook."""

from notebook_cells import ROOT, NotebookBuilder


def main():
    nb = NotebookBuilder()

    nb.markdown(
        """# MAPM: Azure predictive-maintenance tables

Load the five Microsoft sample tables, inspect telemetry for one machine,
and plot registered Plotly views.

[Provider](https://github.com/microsoft/sqlworkshops)
· [Dataset guide](../../pdmdata/datasets/mapm/README.md)

Download explicitly first with `pdmdata.download("mapm")` if local files are
missing. Treat the data as synthetic.
""",
    )
    nb.code(
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
    nb.markdown(
        """## One machine trajectory

Filter telemetry and errors for a single `machineID`, then reuse the shared
Plotly registry.
""",
    )
    nb.code(
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

    nb.write(ROOT / "notebooks/datasets/mapm.ipynb")


if __name__ == "__main__":
    main()
