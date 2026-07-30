"""CARE to Compare visualization helpers."""

from typing import Optional, Sequence

import polars as pl

from datasets.visualization.models import AxisSpec, TimeSeriesSpec
from datasets.visualization.plots import plot_large_time_series


TIME_CANDIDATES = ("timestamp", "time", "datetime", "DateTime")


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
        columns = [
            name
            for name, dtype in schema.items()
            if name != time_column and dtype.is_numeric()
        ][:4]
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
