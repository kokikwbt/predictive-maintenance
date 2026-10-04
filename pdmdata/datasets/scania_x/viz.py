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
    values = frame.select(columns).to_numpy().T
    figure, axis = plt.subplots(figsize=(12, 4), layout="constrained")
    image = axis.imshow(
        values,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(
            frame["time_step"][0],
            frame["time_step"][-1],
            -0.5,
            len(columns) - 0.5,
        ),
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
