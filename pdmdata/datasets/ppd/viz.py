"""PPD plotting helpers that retain recording boundaries and missing values."""

import matplotlib.pyplot as plt
import numpy as np
from .loader import SENSOR_COLUMNS

DEFAULT_SIGNALS = ["L_1", "L_2", "A_1", "B_1", "C_1", "A_5"]


def plot_waveforms(frame, *, columns=None, start=0, stop=None):
    """Plot selected source signals against continuous sequence indices without filling gaps. Source-file boundaries are marked."""
    columns = DEFAULT_SIGNALS if columns is None else list(columns)
    if not columns or any(c not in SENSOR_COLUMNS for c in columns):
        raise ValueError("Select at least one PPD process signal")
    stop = frame.height if stop is None else stop
    if not 0 <= start < stop <= frame.height or stop - start < 2:
        raise ValueError("Select at least two rows within the recording")
    if frame["sequence_id"].n_unique() != 1:
        raise ValueError("Plot one sequence at a time")
    selected = frame.slice(start, stop - start)
    x = selected["sample"].to_numpy()
    figure, axes = plt.subplots(len(columns), 1, sharex=True, squeeze=False,
                                figsize=(12, 1.4 * len(columns) + 1), layout="constrained")
    for axis, column in zip(axes.flat, columns):
        y = selected[column].to_numpy().copy()
        # Nulls become NaN; also break lines if a source index is skipped.
        y[np.r_[False, np.diff(x) > 1]] = np.nan
        axis.plot(x, y, linewidth=.7, color="#225ea8")
        # A dashed marker identifies each joined part without inventing downtime.
        boundary = frame.filter(frame["source_file"] != frame["source_file"].shift(1))["sample"]
        for index in boundary:
            if x[0] <= index <= x[-1]:
                axis.axvline(index, color="#777777", linestyle="--", linewidth=.7)
        axis.set_ylabel(column)
        axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel("Sequence sample index")
    figure.suptitle(f"PPD: sequence C{frame['sequence_id'][0]} | source values")
    return figure


def plot_recordings(recordings, *, column="L_1", stop=1500):
    """Compare separate trial examples without creating a continuous timeline."""
    if column not in SENSOR_COLUMNS or not recordings or stop < 2:
        raise ValueError("Provide recordings, a process signal, and at least two samples")
    figure, axes = plt.subplots((len(recordings) + 1)//2, 2, sharey=True,
                                figsize=(12, 2.2 * ((len(recordings)+1)//2)),
                                squeeze=False, layout="constrained")
    for axis, frame in zip(axes.flat, recordings):
        selected = frame.head(stop)
        axis.plot(selected["sample"], selected[column], linewidth=.7)
        axis.set(title=f"C{frame['sequence_id'][0]}", xlabel="Sequence sample index", ylabel=column)
        axis.grid(alpha=.2)
    for axis in list(axes.flat)[len(recordings):]:
        axis.set_visible(False)
    figure.suptitle(f"PPD: independent sequence prefixes | {column} in source units")
    return figure
