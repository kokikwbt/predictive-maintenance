"""Compact IMS waveform comparisons and recording-level RMS histories."""

import matplotlib.pyplot as plt
from .loader import channel_info


def plot_waveforms(examples, *, experiment=2, stop_s=0.05):
    """Compare short snapshots with shared amplitude limits for each channel."""
    if not examples or not 0 < stop_s <= 1.024:
        raise ValueError("Provide examples and a stop_s in (0, 1.024]")
    info = channel_info(experiment)
    channels = [c for c in info["channel"] if c in examples[0][1].columns]
    if not channels:
        raise ValueError("No vibration channels to plot")
    figure, axes = plt.subplots(len(channels), len(examples), sharex=True,
                                sharey="row", squeeze=False,
                                figsize=(5.5 * len(examples), 2 * len(channels)),
                                layout="constrained")
    for column, (label, frame) in enumerate(examples):
        window = frame.filter(frame["time_s"] < stop_s)
        for index, channel in enumerate(channels):
            axis = axes[index, column]
            bearing = info.filter(info["channel"] == channel)["bearing"][0]
            axis.plot(window["time_s"] * 1000, window[channel], linewidth=.6)
            axis.set_title(f"{label} | bearing {bearing}, {channel}", fontsize=9)
            axis.set_xlabel("Nominal elapsed time (ms)")
            if column == 0:
                axis.set_ylabel("Source amplitude")
            axis.grid(alpha=.2)
    figure.suptitle(f"IMS experiment {experiment}: first {stop_s * 1000:g} ms of selected snapshots")
    return figure


def plot_rms(history, *, experiment=2):
    """Show snapshot RMS against elapsed days, preserving acquisition gaps."""
    info = channel_info(experiment)
    figure, axes = plt.subplots(info.height, 1, sharex=True, squeeze=False,
                                figsize=(12, 1.7 * info.height), layout="constrained")
    elapsed = (history["recording_time"] - history["recording_time"][0]).dt.total_seconds() / 86400
    for axis, row in zip(axes.flat, info.iter_rows(named=True)):
        channel = row["channel"]
        axis.scatter(elapsed, history[channel], s=5)
        axis.set_ylabel(f"{channel} RMS")
        axis.set_title(f"Bearing {row['bearing']} | end-of-test fault: {row['end_of_test_fault']}", fontsize=9)
        axis.grid(alpha=.2)
    axes[-1, 0].set_xlabel("Elapsed days from first selected filename timestamp")
    figure.suptitle(f"IMS experiment {experiment}: RMS per snapshot (source amplitude units)")
    return figure
