"""SCANIA Component X visualization configuration."""

from pdmdata.visualization.models import AxisSpec, TimeSeriesSpec


PLOTS = {
    "counter_trajectory": TimeSeriesSpec(
        title="SCANIA Component X numerical counters",
        x=AxisSpec("time_step", "Anonymized operating time", "time_step"),
        y=(
            AxisSpec("171_0", "Counter 171_0"),
            AxisSpec("666_0", "Counter 666_0"),
            AxisSpec("837_0", "Counter 837_0"),
            AxisSpec("309_0", "Counter 309_0"),
        ),
        entity_column="vehicle_id",
    )
}


def plot_histogram_evolution(frame, *, variable="459"):
    """Show how one histogram variable's bins accumulate for one vehicle."""
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from .loader import histogram_columns

    if isinstance(frame, pl.LazyFrame):
        raise ValueError("Collect one vehicle's readouts before plotting")
    if frame["vehicle_id"].n_unique() != 1:
        raise ValueError("Select exactly one SCANIA Component X vehicle")
    columns = histogram_columns(variable)
    if set(columns) - set(frame.columns):
        raise ValueError("The frame lacks histogram {} columns".format(variable))
    frame = frame.sort("time_step")
    if frame.height < 2:
        raise ValueError("At least two readouts are required")
    values = frame.select(columns).to_numpy().T
    times = frame["time_step"].to_numpy()
    # Readouts are irregular, so cell edges sit midway between readouts.
    edges = np.concatenate(
        [times[:1], (times[1:] + times[:-1]) / 2, times[-1:]]
    )
    figure, axis = plt.subplots(figsize=(12, 4), layout="constrained")
    image = axis.pcolormesh(
        edges, np.arange(len(columns) + 1) - 0.5, values, shading="flat"
    )
    axis.set(
        xlabel="Anonymized operating time (time_step)",
        ylabel="Bin index",
        title="Vehicle {}: histogram {} by readout".format(
            frame["vehicle_id"][0], variable
        ),
    )
    figure.colorbar(image, ax=axis, label="Source value")
    return figure
