"""Matplotlib views of GDD motor signals and categorical state intervals."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from .preprocessing import MOTOR_SIGNALS, state_segments


def plot_states(frame, *, start=0, stop=800):
    """Plot source-order samples with state colors; stop is exclusive."""
    if not 0 <= start < stop <= frame.height:
        raise ValueError("Require 0 <= start < stop <= number of observations")
    labels = sorted(frame["Label"].unique().to_list())
    colors = plt.get_cmap("tab10")(np.arange(len(labels)))
    segments = state_segments(frame)
    figure, axes = plt.subplots(6, 1, sharex=True, figsize=(12, 9),
                                gridspec_kw={"height_ratios": [1, 2, 2, 2, 2, 2]},
                                layout="constrained")
    window = frame.slice(start, stop-start)
    codes = {label:i for i,label in enumerate(labels)}
    mesh = axes[0].pcolormesh(np.arange(start,stop+1), [0,1],
                              [[codes[v] for v in window["Label"]]],
                              cmap=ListedColormap(colors),
                              norm=BoundaryNorm(np.arange(len(labels)+1)-.5,len(labels)),
                              shading="flat", rasterized=True)
    axes[0].set_yticks([])
    axes[0].set_ylabel("State")
    bar = figure.colorbar(mesh, ax=list(axes), ticks=range(len(labels)), fraction=.025, pad=.02)
    bar.set_ticklabels([str(v) for v in labels])
    bar.set_label("Label ID")
    for column, axis in zip(MOTOR_SIGNALS, axes[1:]):
        axis.plot(np.arange(start,stop), window[column].to_numpy(), color="#172b4d", linewidth=.8)
        for row in segments.iter_rows(named=True):
            a,b=max(start,row['start_sample']),min(stop,row['end_sample'])
            if a<b:axis.axvspan(a,b,color=colors[codes[row['Label']]],alpha=.14,linewidth=0)
        axis.set_ylabel(column.removeprefix('MotorData.'),fontsize=9)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Sample index (original file order)")
    axes[-1].set_xlim(start,stop)
    figure.suptitle(f"GDD state transitions and motor signals · samples {start}–{stop-1}")
    return figure
