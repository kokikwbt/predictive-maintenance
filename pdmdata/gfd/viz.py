"""GFD visualization configuration."""

from pdmdata.visualization.models import AxisSpec, DistributionSpec


PLOTS = {
    "condition_distribution": DistributionSpec(
        title="Gearbox vibration by condition",
        feature=AxisSpec("sensor_1", "Vibration sensor 1"),
        group_column="condition",
        group_label="Gear condition",
    )
}


def plot_waveforms(healthy, broken, *, start=0, stop=2048):
    """Compare independent recordings at one load using identical channel scales.

    The x-axis is sample index: a sampling rate is not established by the raw
    files. The two recordings are not synchronized or concatenated.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from .loader import SENSOR_COLUMNS

    if not 0 <= start < stop <= min(healthy.height, broken.height):
        raise ValueError("Select a nonempty sample range within both recordings")
    levels = []
    for frame, condition in ((healthy, "healthy"), (broken, "broken")):
        if frame["condition"].unique().to_list() != [condition]:
            raise ValueError(f"Expected a single {condition} recording")
        if frame["load"].n_unique() != 1:
            raise ValueError("Each frame must contain exactly one load level")
        levels.append(frame["load"][0])
    if levels[0] != levels[1]:
        raise ValueError("Compare healthy and broken recordings at the same load")
    figure, axes = plt.subplots(4, 2, sharex=True, sharey="row",
                                figsize=(12, 8), layout="constrained")
    for j, (frame, condition, color) in enumerate(
        ((healthy, "Healthy", "#225ea8"), (broken, "Broken tooth", "#d95f0e"))
    ):
        selected = frame.slice(start, stop - start)
        for i, column in enumerate(SENSOR_COLUMNS):
            axes[i, j].plot(np.arange(start, stop), selected[column],
                            linewidth=.6, color=color)
            axes[i, j].grid(alpha=.2)
            if j == 0:
                axes[i, j].set_ylabel(f"{column}\nSource amplitude")
        axes[0, j].set_title(condition)
        axes[-1, j].set_xlabel("Sample index (within each recording)")
    figure.suptitle(f"GFD: independent recordings at {levels[0]}% load")
    return figure


def plot_rms(stats):
    """Plot per-recording RMS summaries against load, without implying continuity."""
    import matplotlib.pyplot as plt
    import polars as pl
    from .loader import SENSOR_COLUMNS

    figure, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for axis, sensor in zip(axes.flat, SENSOR_COLUMNS):
        for condition, color in [("healthy", "#225ea8"), ("broken", "#d95f0e")]:
            data = stats.filter(
                (pl.col("sensor") == sensor) & (pl.col("condition") == condition)
            ).sort("load")
            axis.scatter(data["load"], data["rms"], label=condition, color=color)
        axis.set(title=sensor, xlabel="Load (%)", ylabel="RMS (source amplitude)",
                 xticks=range(0, 100, 20))
        axis.grid(alpha=.2)
    axes[0, 0].legend()
    figure.suptitle("GFD: separate recordings at each load")
    return figure


def plot_correlations(healthy, broken):
    """Compare marginal sensor correlations; these are not precision matrices."""
    import matplotlib.pyplot as plt
    import numpy as np
    from .loader import SENSOR_COLUMNS

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for axis, (condition, frame) in zip(axes, [("Healthy", healthy), ("Broken tooth", broken)]):
        values = frame.select(SENSOR_COLUMNS).to_numpy()
        if not np.all(values.std(axis=0) > 0):
            raise ValueError("Inspect constant channels before correlation")
        correlation = np.corrcoef(values, rowvar=False)
        image = axis.imshow(correlation, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set(xticks=range(4), yticks=range(4), xticklabels=SENSOR_COLUMNS,
                 yticklabels=SENSOR_COLUMNS, title=f"{condition}, {frame['load'][0]}% load")
        axis.tick_params(axis="x", labelrotation=30)
        for i in range(4):
            for j in range(4):
                axis.text(j, i, f"{correlation[i,j]:.2f}", ha="center", va="center",
                          color="white" if abs(correlation[i,j]) > .6 else "black")
    figure.colorbar(image, ax=list(axes), label="Pearson correlation")
    return figure
