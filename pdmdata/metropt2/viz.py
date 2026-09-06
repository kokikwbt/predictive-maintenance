"""MetroPT2 visualization configuration."""

from pdmdata.visualization.models import AxisSpec, TimeSeriesSpec
from pdmdata.visualization.plots import plot_large_time_series


SENSOR_SPEC = TimeSeriesSpec(
    title="MetroPT2 air-production-unit signals",
    x=AxisSpec("timestamp", "Timestamp"),
    y=(
        AxisSpec("TP2", "Compressor pressure", "bar"),
        AxisSpec("TP3", "Pneumatic-panel pressure", "bar"),
        AxisSpec("Oil_temperature", "Oil temperature", "°C"),
        AxisSpec("Motor_current", "Motor current", "A"),
    ),
    max_points=10_000,
)


def sensor_signals(frame, **options):
    """Plot a bounded set of the principal MetroPT2 analogue signals."""
    return plot_large_time_series(frame, SENSOR_SPEC, **options)


PLOTS = {"sensor_signals": sensor_signals}


ANALOG_SIGNALS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs",
                  "Oil_temperature", "Flowmeter", "Motor_current"]
CONTROL_SIGNALS = ["COMP", "DV_eletric", "Towers"]


def plot_operation(frame, *, columns=None):
    """Plot actual timestamps, control flags, and analogue signals with Matplotlib.

    Control flags are reference signals, not annotated operating-state labels.
    Callers must select a manageable time interval before passing the frame.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    columns = list(columns) if columns is not None else [
        "TP2", "TP3", "Reservoirs", "Motor_current", "Oil_temperature", "Flowmeter"]
    if not columns:
        raise ValueError("Select at least one analogue signal")
    if any(c not in ANALOG_SIGNALS for c in columns):
        raise ValueError("Select columns from the MetroPT2 analogue signals")
    selected = frame.select(["timestamp", *CONTROL_SIGNALS, *columns])
    if isinstance(selected, pl.LazyFrame):
        selected = selected.collect()
    if selected.height < 2:
        raise ValueError("At least two observations are required")
    if not selected["timestamp"].is_sorted():
        raise ValueError("Timestamps are not ordered; inspect the source before plotting")
    seconds = (selected["timestamp"] - selected["timestamp"][0]).dt.total_milliseconds().to_numpy()/1000
    x = seconds/60
    # Show gaps in analogue traces instead of interpolating across outages.
    gaps = np.r_[False, np.diff(seconds) > 5]
    figure, axes = plt.subplots(len(columns)+1, 1, sharex=True,
                                figsize=(12, 2 + 1.5*len(columns)), layout="constrained")
    for i,c in enumerate(CONTROL_SIGNALS):
        y=selected[c].cast(pl.Float64).to_numpy().copy()
        y[gaps]=np.nan
        axes[0].step(x,y*.6+i,where="post",linewidth=.9,label=c)
    axes[0].set_yticks([.3+i for i in range(len(CONTROL_SIGNALS))], CONTROL_SIGNALS)
    axes[0].set_ylabel("Binary controls\n(offset for display)")
    for c,axis in zip(columns,axes[1:]):
        y=selected[c].cast(pl.Float64).to_numpy().copy()
        y[gaps]=np.nan
        axis.plot(x,y,linewidth=.7,color="#225ea8")
        axis.set_ylabel(c)
        axis.grid(alpha=.2)
    axes[-1].set_xlabel(f"Minutes since {selected['timestamp'][0]}")
    axes[-1].set_xlim(x[0],x[-1])
    figure.suptitle("MetroPT2: control switching and sensor waveforms")
    return figure
