"""Generate an unexecuted SCANIA Component X exploration notebook."""

from notebook_cells import ROOT, NotebookBuilder


def main():
    nb = NotebookBuilder()

    nb.markdown(
        """# SCANIA Component X: readouts and repair labels

Inspect train/validation inventory, load one vehicle with TTE targets, and
plot counter trajectories.

[Dataset guide](../../pdmdata/datasets/scania_x/README.md)

Download explicitly first with `pdmdata.download("scania_x")`. The challenge
files are large and excluded from bulk download.
""",
    )
    nb.code(
        """from pathlib import Path
import sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pdmdata").is_dir())
sys.path.insert(0, str(ROOT))
from IPython.display import display
import pdmdata
from pdmdata.datasets.scania_x import inventory, vehicles

display(inventory())
train_vehicles = vehicles("train")
print(f"Train vehicles: {train_vehicles.height}")
VEHICLE = int(train_vehicles["vehicle_id"][0])
labeled = pdmdata.load(
    "scania_x",
    split="train",
    vehicle_id=VEHICLE,
    with_labels=True,
    lazy=False,
)
display(labeled.select(
    ["vehicle_id", "time_step", "time_to_event", "event_observed"]
).head())
figure = pdmdata.visualize(
    "scania_x", "counter_trajectory", labeled, entity=VEHICLE
)
figure.show()
""",
    )

    nb.write(ROOT / "notebooks/datasets/scania_x.ipynb")


if __name__ == "__main__":
    main()
