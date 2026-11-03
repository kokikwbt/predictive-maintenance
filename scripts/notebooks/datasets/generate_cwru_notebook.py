"""Generate the CWRU discovery, source loading and visualization notebook."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from notebook_cells import ROOT, NotebookBuilder  # noqa: E402


def main() -> None:
    nb = NotebookBuilder()

    nb.markdown(
        """# CWRU: normal and faulty bearing vibration

Explore all four official collections for classification, anomaly detection,
signal analysis and condition adaptation. Split recordings before windowing.

[Dataset API guide](../../pdmdata/datasets/cwru/README.md)
"""
    )
    nb.code(
        """from IPython.display import display

import pdmdata
from pdmdata.datasets import cwru

display(cwru.inventory().group_by("collection").len().sort("collection"))
display(cwru.inventory(collection="12k_de", fault="inner_race", load_hp=[0, 1]))
display(cwru.inventory(installed=True))
"""
    )
    nb.markdown(
        """## Explicit bounded acquisition

Enable this switch to download two selected records. Existing source files are
reused. Loading never downloads automatically.
"""
    )
    nb.code(
        """DOWNLOAD = False
if DOWNLOAD:
    cwru.download(recording=[97, 105], max_bytes=32 * 1024**2)
"""
    )
    nb.code(
        """RECORDING = 105
if cwru.inventory(recording=RECORDING, installed=True).is_empty():
    raise FileNotFoundError("Download recording 105 explicitly before continuing")
display(cwru.variables(RECORDING))
display(cwru.channel_info(RECORDING))
signal = pdmdata.load("cwru", recording=RECORDING, channels="all")
display(signal.head())
pdmdata.visualize("cwru", "waveform", signal).show()
pdmdata.visualize("cwru", "spectrum", signal, channel="DE", nperseg=4096).show()
"""
    )
    nb.markdown(
        """## Normal baseline and raw variables

Normal rates are unspecified in the adapter. Sample indices and amplitudes
remain usable; a spectrum needs an independently established rate. Record 99
also contains X098 arrays. variables() exposes these, while load() selects X099.
"""
    )
    nb.code(
        """if not cwru.inventory(recording=97, installed=True).is_empty():
    baseline = cwru.load(97, channels="all")
    display(cwru.channel_info(97))
    pdmdata.visualize("cwru", "waveform", baseline).show()
if not cwru.inventory(recording=99, installed=True).is_empty():
    display(cwru.variables(99))
    print("Embedded array shape:", cwru.load_variable(99, "X098_DE_time").shape)
display(cwru.verify(recording=RECORDING))
"""
    )
    nb.markdown(
        """## Keep recordings separate

iter_load() processes selected recordings one at a time. A sample slice does
not apply an anti-alias filter. Fault onset, RUL and maintenance outcomes are
not observed in these experiments.
"""
    )
    nb.code(
        """for frame in cwru.iter_load(recording=RECORDING, channels="DE", start=1000, stop=5000):
    display(frame.select("recording", "channel", "sample", "amplitude").head())
"""
    )

    nb.write(ROOT / "notebooks/datasets/cwru.ipynb")


if __name__ == "__main__":
    main()
