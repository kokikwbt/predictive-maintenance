"""CARE to Compare visualization helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import polars as pl

from pdmdata.visualization.models import AxisSpec, TimeSeriesSpec
from pdmdata.visualization.plots import plot_large_time_series


TIME_CANDIDATES = ("time_stamp", "timestamp", "time", "datetime", "DateTime")


def scada_signals(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    time_column: Optional[str] = None,
):
    """Plot selected numeric signals from one CARE recording."""
    names = frame.collect_schema().names() if isinstance(frame, pl.LazyFrame) else frame.columns
    time_column = time_column or next(
        (candidate for candidate in TIME_CANDIDATES if candidate in names),
        None,
    )
    if time_column is None:
        raise ValueError(
            "CARE timestamp column was not detected; pass time_column explicitly"
        )
    if columns is None:
        schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
        numeric = [
            name
            for name, dtype in schema.items()
            if name not in {time_column, "asset_id", "id", "status_type_id", "event_id"}
            and dtype.is_numeric()
        ]
        columns = ([name for name in numeric if name.endswith("_avg")] or numeric)[:4]
    if not columns:
        raise ValueError("No CARE signal columns were selected")
    spec = TimeSeriesSpec(
        title="CARE wind-turbine SCADA signals",
        x=AxisSpec(time_column, "Timestamp"),
        y=tuple(AxisSpec(column, column.replace("_", " ")) for column in columns),
        max_points=10_000,
    )
    return plot_large_time_series(frame, spec)


PLOTS = {"scada_signals": scada_signals}


def plot_waveforms(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    columns: Optional[Sequence[str]] = None,
    title: str = "CARE SCADA waveforms",
    event: Optional[dict] = None,
):
    """Plot full-resolution CARE signals on separate Matplotlib axes.

    Only selected columns are collected, over the entire supplied time range.
    Prediction and labeled event intervals are shaded without changing raw data.
    """
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    if "time_stamp" not in schema:
        raise ValueError("CARE waveforms require the time_stamp column")
    if columns is None:
        columns = [
            name for name, dtype in schema.items()
            if name.endswith("_avg") and dtype.is_numeric()
        ][:4]
    columns = list(columns)
    if not columns:
        raise ValueError("Select at least one numeric CARE signal column")
    for column in columns:
        if column not in schema or not schema[column].is_numeric():
            raise ValueError(f"Unknown or non-numeric CARE signal: {column}")
    selected_columns = list(dict.fromkeys(
        ["time_stamp", *columns] + (["train_test"] if "train_test" in schema else [])
    ))
    selected = frame.select(selected_columns).drop_nulls("time_stamp").sort("time_stamp")
    if isinstance(selected, pl.LazyFrame):
        selected = selected.collect()
    if selected.is_empty():
        raise ValueError("No CARE observations remain in the selected time range")
    times = selected["time_stamp"].to_list()
    prediction = None
    if "train_test" in selected.columns:
        prediction_times = selected.filter(pl.col("train_test") == "prediction")["time_stamp"]
        if len(prediction_times):
            prediction = (prediction_times.min(), prediction_times.max())

    figure, axes = plt.subplots(
        len(columns), 1, sharex=True, squeeze=False,
        figsize=(13, 2.5 * len(columns)), layout="constrained",
    )
    for column, axis in zip(columns, axes[:, 0]):
        axis.plot(times, selected[column].to_numpy(), color="#2563a6", linewidth=0.6)
        if prediction is not None:
            axis.axvspan(*prediction, color="#f4b942", alpha=0.18, label="Prediction interval")
        if event and event.get("event_start") is not None and event.get("event_end") is not None:
            start = max(times[0], event["event_start"])
            end = min(times[-1], event["event_end"])
            if start < end:
                axis.axvspan(start, end, color="#d74b46", alpha=0.18,
                            label=f"Event interval ({event.get('event_label', 'event')})")
        axis.set_ylabel(column, fontsize=9)
        axis.grid(alpha=0.2)
        if len(times) > 1:
            axis.set_xlim(times[0], times[-1])
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axes[-1, 0].set_xlabel("Time (anonymized)")
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(loc="upper left", fontsize=8)
    figure.suptitle(title)
    return figure
