"""C-MAPSS visualization configuration."""

from pdmdata.visualization.models import AxisSpec, TimeSeriesSpec


PLOTS = {
    "sensor_trajectory": TimeSeriesSpec(
        title="C-MAPSS sensor trajectory",
        x=AxisSpec("cycle", "Operating cycle"),
        y=(
            AxisSpec("sensor_2", "Sensor 2"),
            AxisSpec("sensor_7", "Sensor 7"),
            AxisSpec("sensor_11", "Sensor 11"),
            AxisSpec("sensor_12", "Sensor 12"),
        ),
        entity_column="unit_number",
    )
}


DEFAULT_SENSORS = ["sensor_2", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15"]


def plot_waveforms(frame, *, subset="FD001", split="train", sensors=None, stop_cycle=None):
    """Plot one engine's source measurements without smoothing or resampling."""
    import matplotlib.pyplot as plt
    if frame["unit_number"].n_unique() != 1:
        raise ValueError("Select exactly one C-MAPSS unit before plotting")
    sensors = DEFAULT_SENSORS if sensors is None else list(sensors)
    if not sensors or set(sensors) - set(frame.columns):
        raise ValueError("Select available sensor columns")
    if stop_cycle is not None:
        frame = frame.filter(frame["cycle"] <= stop_cycle)
    if frame.is_empty():
        raise ValueError("No cycles in the selected range")
    figure, axes = plt.subplots((len(sensors) + 1) // 2, 2, sharex=True, squeeze=False,
                                figsize=(12, 2.25 * ((len(sensors) + 1) // 2)), layout="constrained")
    for axis, sensor in zip(axes.flat, sensors):
        axis.plot(frame["cycle"], frame[sensor], linewidth=.8)
        axis.set(title=sensor, ylabel="Source value", xlabel="Operating cycle")
        axis.grid(alpha=.2)
    for axis in list(axes.flat)[len(sensors):]:
        axis.set_visible(False)
    figure.suptitle(f"C-MAPSS {subset} {split}, unit {frame['unit_number'][0]}")
    return figure


def plot_operating_settings(frame, *, subset="FD002", stop_cycle=80):
    """Align the three recorded settings with two sensors for one engine."""
    import matplotlib.pyplot as plt
    if frame["unit_number"].n_unique() != 1:
        raise ValueError("Select exactly one C-MAPSS unit before plotting")
    frame = frame.filter(frame["cycle"] <= stop_cycle)
    if frame.is_empty():
        raise ValueError("No cycles in the selected range")
    columns = ["operation_1", "operation_2", "operation_3", "sensor_2", "sensor_11"]
    figure, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 8), layout="constrained")
    for axis, column in zip(axes, columns):
        axis.plot(frame["cycle"], frame[column], marker=".", markersize=3, linewidth=.7)
        axis.set_ylabel(column)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Operating cycle")
    figure.suptitle(f"C-MAPSS {subset}, unit {frame['unit_number'][0]}: operating settings and sensor response")
    return figure


def plot_rul(train, test, *, subset="FD001"):
    """Show targets for separate train/test engines without joining their lives."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.5), layout="constrained")
    for axis, frame, split in zip(axes, [train, test], ["train", "test"]):
        if frame["unit_number"].n_unique() != 1 or "RUL" not in frame.columns:
            raise ValueError("Select one engine with RUL in each split")
        axis.plot(frame["cycle"], frame["RUL"])
        axis.scatter(frame["cycle"][-1], frame["RUL"][-1], color="tab:red", s=25,
                     label=f"Last observed RUL: {frame['RUL'][-1]} cycles")
        axis.set(title=f"{subset} {split}, unit {frame['unit_number'][0]}",
                 xlabel="Observed operating cycle", ylabel="RUL (cycles)")
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    figure.suptitle("Distinct train/test engines: uncapped RUL targets")
    return figure
