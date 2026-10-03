"""Matplotlib views of OYICD recordings and operating-mode coverage."""

import numpy as np
import matplotlib.pyplot as plt
from .loader import SENSOR_COLUMNS

SIGNAL_LABELS = [
    "Cut motor torque", "Cut position lag error", "Cut actual position",
    "Cut actual speed", "Film actual position", "Film actual speed",
    "Film position lag error", "Spintor VAX speed",
]


def plot_waveforms(frame, *, start=0, stop=None):
    """Show eight source signals against elapsed seconds within one recording."""
    stop = frame.height if stop is None else stop
    if not 0 <= start < stop <= frame.height or stop - start < 2:
        raise ValueError("Select at least two samples within the recording")
    if frame["filename"].n_unique() != 1:
        raise ValueError("Pass one recording; do not concatenate independent captures")
    selected = frame.slice(start, stop - start)
    time = selected["timestamp"].to_numpy() - frame["timestamp"][0]
    figure, axes = plt.subplots(4, 2, sharex=True, figsize=(12, 9), layout="constrained")
    for axis, column, label in zip(axes.flat, SENSOR_COLUMNS, SIGNAL_LABELS):
        axis.plot(time, selected[column], linewidth=.8, color="#225ea8")
        axis.set(title=label, ylabel="Source value")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel("Seconds since first observation")
    figure.suptitle(f"OYICD: {frame['filename'][0]}")
    return figure


def plot_coverage(files):
    """Count unique recordings by relative month and filename mode."""
    counts = np.zeros((8, 12), dtype=int)
    for row in files.iter_rows(named=True):
        counts[row["mode"] - 1, row["month"] - 1] += 1
    figure, axis = plt.subplots(figsize=(10, 4), layout="constrained")
    image = axis.imshow(counts, cmap="Blues", aspect="auto", vmin=0)
    for i in range(8):
        for j in range(12):
            axis.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=8,
                      color="white" if counts[i, j] > counts.max()/2 else "black")
    axis.set(xticks=range(12), xticklabels=range(1, 13), yticks=range(8), yticklabels=range(1, 9),
             xlabel="Relative month", ylabel="Operating mode", title="OYICD: unique recording coverage")
    figure.colorbar(image, ax=axis, label="Recordings")
    return figure


def plot_modes(recordings):
    """Compare torque examples from separate modes on shared amplitude scales."""
    figure, axes = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(12, 9), layout="constrained")
    if len(recordings) != 8 or sorted(f['mode'][0] for f in recordings) != list(range(1, 9)):
        raise ValueError("Provide one recording for each of the eight modes")
    for axis, frame in zip(axes.flat, sorted(recordings, key=lambda f: f['mode'][0])):
        axis.plot(frame["timestamp"] - frame["timestamp"][0], frame[SENSOR_COLUMNS[0]], linewidth=.8)
        axis.set_title(f"Mode {frame['mode'][0]} | {frame['filename'][0]}", fontsize=9)
        axis.set_ylabel("Motor torque (source value)")
        axis.grid(alpha=.2)
    for axis in axes[-1]:
        axis.set_xlabel("Seconds since first observation")
    figure.suptitle("OYICD: independent captures, one example per mode")
    return figure
