"""Generate an unexecuted N-CMAPSS exploration notebook."""

from notebook_cells import ROOT, NotebookBuilder


def main():
    nb = NotebookBuilder()

    nb.markdown(
        """# N-CMAPSS: run-to-failure trajectories

Inventory a local variant, load one unit with per-cycle RUL, and plot sensors
through the shared Plotly API. Rows are 1 Hz samples within each flight
(`cycle`), so a full split has millions of rows; load one unit at a time.

[Dataset guide](../../pdmdata/datasets/ncmapss/README.md)

Download a variant explicitly first, for example
`pdmdata.download("ncmapss", variant="ds01")`. Large files are opt-in and
excluded from bulk download.
""",
    )
    nb.code(
        """from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))
from IPython.display import display
import pdmdata
from pdmdata.datasets.ncmapss import available_variants, inventory, units

print("Variants:", available_variants())
summary = inventory()
display(summary)
VARIANT = "ds01"
dev_units = units(VARIANT, "dev")
print(f"{VARIANT} dev units: {dev_units}")
UNIT = dev_units[0]
unit = pdmdata.load(
    "ncmapss", variant=VARIANT, split="dev", unit=UNIT, with_rul=True
)
print(f"Unit {UNIT}: {unit.height:,} rows over {unit['cycle'].n_unique()} cycles")
display(unit.select(["unit_number", "cycle", "RUL"]).head())
figure = pdmdata.visualize(
    "ncmapss", "sensor_trajectory", unit, entity=UNIT
)
figure.show()
""",
    )

    nb.write(ROOT / "notebooks/datasets/ncmapss.ipynb")


if __name__ == "__main__":
    main()
