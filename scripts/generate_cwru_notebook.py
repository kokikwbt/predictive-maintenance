"""Generate the CWRU discovery, source loading and visualization notebook."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [
        (
            "markdown",
            """# CWRU: normal and faulty bearing vibration

Explore all four official collections for classification, anomaly detection,
signal analysis and condition adaptation. Split recordings before windowing.

[Dataset API guide](../../pdmdata/datasets/cwru/README.md)
""",
        ),
        (
            "code",
            """from IPython.display import display

import pdmdata
from pdmdata.datasets import cwru

display(cwru.inventory().group_by("collection").len().sort("collection"))
display(cwru.inventory(collection="12k_de", fault="inner_race", load_hp=[0, 1]))
display(cwru.inventory(installed=True))
""",
        ),
        (
            "markdown",
            """## Explicit bounded acquisition

Enable this switch to download two selected records. Existing source files are
reused. Loading never downloads automatically.
""",
        ),
        (
            "code",
            """DOWNLOAD = False
if DOWNLOAD:
    cwru.download(recording=[97, 105], max_bytes=32 * 1024**2)
""",
        ),
        (
            "code",
            """RECORDING = 105
if cwru.inventory(recording=RECORDING, installed=True).is_empty():
    raise FileNotFoundError("Download recording 105 explicitly before continuing")
display(cwru.variables(RECORDING))
display(cwru.channel_info(RECORDING))
signal = pdmdata.load("cwru", recording=RECORDING, channels="all")
display(signal.head())
pdmdata.visualize("cwru", "waveform", signal).show()
pdmdata.visualize("cwru", "spectrum", signal, channel="DE", nperseg=4096).show()
""",
        ),
        (
            "markdown",
            """## Normal baseline and raw variables

Normal rates are unspecified in the adapter. Sample indices and amplitudes
remain usable; a spectrum needs an independently established rate. Record 99
also contains X098 arrays. variables() exposes these, while load() selects X099.
""",
        ),
        (
            "code",
            """if not cwru.inventory(recording=97, installed=True).is_empty():
    baseline = cwru.load(97, channels="all")
    display(cwru.channel_info(97))
    pdmdata.visualize("cwru", "waveform", baseline).show()
if not cwru.inventory(recording=99, installed=True).is_empty():
    display(cwru.variables(99))
    print("Embedded array shape:", cwru.load_variable(99, "X098_DE_time").shape)
display(cwru.validate(recording=RECORDING))
""",
        ),
        (
            "markdown",
            """## Keep recordings separate

iter_load() processes selected recordings one at a time. A sample slice does
not apply an anti-alias filter. Fault onset, RUL and maintenance outcomes are
not observed in these experiments.
""",
        ),
        (
            "code",
            """for frame in cwru.iter_load(recording=RECORDING, channels="DE", start=1000, stop=5000):
    display(frame.select("recording", "channel", "sample", "amplitude").head())
""",
        ),
    ]
    cells = []
    for index, (kind, source) in enumerate(sources):
        cell = {
            "cell_type": kind,
            "id": f"cwru-{index:02d}",
            "metadata": {},
            "source": source.splitlines(True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
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
    }
    path = ROOT / "notebooks" / "datasets" / "cwru.ipynb"
    path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
