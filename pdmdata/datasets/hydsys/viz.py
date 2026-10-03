"""HYDSYS views with native-rate time axes and separate cycle-level targets."""

import matplotlib.pyplot as plt
from .loader import SENSOR_INFO, TARGET_VALUES

DEFAULT_SENSORS = ["PS1", "PS2", "EPS1", "FS1", "TS1", "VS1", "CE", "SE"]
TARGET_LABELS = ["Cooler (%)", "Valve (%)", "Pump leakage code", "Accumulator (bar)", "Stability flag"]


def plot_cycle(groups, *, cycle, sensors=None, start_s=0, stop_s=60):
    """Plot native-rate signals for one cycle without resampling the data."""
    sensors = DEFAULT_SENSORS if sensors is None else list(sensors)
    if not sensors or not 0 <= start_s < stop_s <= 60:
        raise ValueError("Select sensors and an interval within the 60-second cycle")
    figure, axes = plt.subplots((len(sensors)+1)//2, 2, sharex=True, squeeze=False,
                                figsize=(12, 2.2*((len(sensors)+1)//2)), layout="constrained")
    for axis, sensor in zip(axes.flat, sensors):
        rate, unit = SENSOR_INFO[sensor]
        frame = groups[rate]
        selected = frame.filter((frame["time_s"] >= start_s) & (frame["time_s"] < stop_s))
        axis.plot(selected["time_s"], selected[sensor], linewidth=.8,
                  marker="." if rate == 1 else None, markersize=3)
        axis.set(title=f"{sensor} ({rate} Hz)", ylabel=unit, xlabel="Nominal elapsed seconds")
        axis.grid(alpha=.2)
    for axis in list(axes.flat)[len(sensors):]:
        axis.set_visible(False)
    figure.suptitle(f"HYDSYS cycle {cycle}: native sampling rates")
    return figure


def plot_profile(labels):
    """Display cycle-level targets as categorical tracks in source order."""
    figure, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 7), layout="constrained")
    for axis, column, title in zip(axes, TARGET_VALUES, TARGET_LABELS):
        values = sorted(TARGET_VALUES[column])
        lookup = {v: i for i, v in enumerate(values)}
        codes = [lookup[v] for v in labels[column]]
        axis.scatter(labels["cycle"], codes, s=2)
        axis.set_yticks(range(len(values)), labels=values)
        axis.set_ylabel(title)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Cycle index (source order, zero-based)")
    figure.suptitle("HYDSYS: observed condition settings and stability")
    return figure


def plot_valve_comparison(examples):
    """Compare native 100 Hz pressure and power for labeled valve examples."""
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 7), layout="constrained")
    for valve, cycle, frame in examples:
        for axis, sensor in zip(axes, ["PS1", "PS2", "EPS1"]):
            axis.plot(frame["time_s"], frame[sensor], linewidth=.8,
                      label=f"Valve {valve}%, cycle {cycle}")
            axis.set_ylabel(f"{sensor} ({SENSOR_INFO[sensor][1]})")
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("Nominal elapsed seconds within each cycle")
    figure.suptitle("Valve examples: cooler 100%, pump 0, accumulator 130 bar, stable flag 0")
    return figure
