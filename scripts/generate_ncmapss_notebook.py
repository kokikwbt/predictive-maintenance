"""Generate an unexecuted N-CMAPSS exploration notebook."""

from notebook_cells import ROOT, NotebookBuilder


def main():
    nb = NotebookBuilder()

    nb.markdown(
        """# N-CMAPSS: run-to-failure trajectories

Inventory a local variant, load one unit with per-cycle RUL, and plot sensors
through the shared Plotly API.

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

    nb.write(ROOT / "notebooks/datasets/ncmapss.ipynb")


if __name__ == "__main__":
    main()
