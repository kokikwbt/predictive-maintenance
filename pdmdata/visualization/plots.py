"""Reusable Plotly figure builders operating on Polars DataFrames."""

from __future__ import annotations

import math
from typing import Any, Optional

import plotly.graph_objects as go
import polars as pl

from .models import DistributionSpec, EventTimelineSpec, TimeSeriesSpec


def _require_columns(frame: pl.DataFrame, *columns: Optional[str]) -> None:
    missing = [column for column in columns if column and column not in frame.columns]
    if missing:
        raise ValueError("Missing visualization columns: {}".format(", ".join(missing)))


def _downsample(frame: pl.DataFrame, max_points: int) -> pl.DataFrame:
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if frame.height <= max_points:
        return frame
    step = math.ceil(frame.height / max_points)
    return frame.gather_every(step)


def plot_time_series(
    frame: pl.DataFrame,
    spec: TimeSeriesSpec,
    *,
    entity: Optional[Any] = None,
) -> go.Figure:
    """Build an interactive line figure from a time-series specification."""
    _require_columns(
        frame,
        spec.x.column,
        spec.entity_column,
        *(axis.column for axis in spec.y),
    )
    selected = frame
    if spec.entity_column is not None and entity is not None:
        selected = selected.filter(pl.col(spec.entity_column) == entity)
    if selected.is_empty():
        raise ValueError("No rows remain after applying the entity selection")
    selected = _downsample(selected.sort(spec.x.column), spec.max_points)

    figure = go.Figure()
    for axis in spec.y:
        figure.add_trace(
            go.Scattergl(
                x=selected[spec.x.column].to_list(),
                y=selected[axis.column].to_list(),
                mode="lines",
                name=axis.title,
                hovertemplate="%{{x}}<br>%{{y}}<extra>{}</extra>".format(axis.title),
            )
        )
    title = spec.title
    if spec.entity_column is not None and entity is not None:
        title = "{} — {} {}".format(title, spec.entity_column, entity)
    figure.update_layout(
        title=title,
        xaxis_title=spec.x.title,
        yaxis_title="Sensor value",
        hovermode="x unified",
        template="plotly_white",
        legend_title_text="Signal",
    )
    return figure


def plot_large_time_series(
    frame: pl.DataFrame | pl.LazyFrame,
    spec: TimeSeriesSpec,
    *,
    entity: Optional[Any] = None,
) -> go.Figure:
    """Plot a bounded projection without collecting an entire large dataset."""
    if isinstance(frame, pl.DataFrame):
        return plot_time_series(frame, spec, entity=entity)
    columns = [spec.x.column, *(axis.column for axis in spec.y)]
    if spec.entity_column:
        columns.append(spec.entity_column)
    selected = frame
    if spec.entity_column is not None and entity is not None:
        selected = selected.filter(pl.col(spec.entity_column) == entity)
    bounded = (
        selected.select(columns)
        .drop_nulls(subset=[spec.x.column])
        .head(spec.max_points)
        .collect()
    )
    return plot_time_series(bounded, spec, entity=entity)


def plot_distribution(
    frame: pl.DataFrame,
    spec: DistributionSpec,
) -> go.Figure:
    """Build overlaid histograms for a feature grouped by a categorical column."""
    _require_columns(frame, spec.feature.column, spec.group_column)
    figure = go.Figure()
    groups = frame[spec.group_column].drop_nulls().unique().sort().to_list()
    for group in groups:
        values = frame.filter(pl.col(spec.group_column) == group)[
            spec.feature.column
        ].drop_nulls()
        figure.add_trace(
            go.Histogram(
                x=values.to_list(),
                name=str(group),
                nbinsx=spec.bins,
                opacity=0.6,
            )
        )
    figure.update_layout(
        title=spec.title,
        xaxis_title=spec.feature.title,
        yaxis_title="Observations",
        barmode="overlay",
        template="plotly_white",
        legend_title_text=spec.group_label,
    )
    return figure


def plot_event_timeline(
    frame: pl.DataFrame,
    spec: EventTimelineSpec,
    *,
    entity: Optional[Any] = None,
) -> go.Figure:
    """Build an interactive categorical event timeline."""
    _require_columns(
        frame,
        spec.time.column,
        spec.entity_column,
        spec.event_column,
    )
    selected = frame
    if entity is not None:
        selected = selected.filter(pl.col(spec.entity_column) == entity)
    if selected.is_empty():
        raise ValueError("No rows remain after applying the entity selection")
    selected = _downsample(selected.sort(spec.time.column), spec.max_points)

    figure = go.Figure()
    events = selected[spec.event_column].drop_nulls().unique().sort().to_list()
    for event in events:
        event_rows = selected.filter(pl.col(spec.event_column) == event)
        figure.add_trace(
            go.Scattergl(
                x=event_rows[spec.time.column].to_list(),
                y=event_rows[spec.entity_column].to_list(),
                mode="markers",
                name=str(event),
                marker={"size": 8},
            )
        )
    figure.update_layout(
        title=spec.title,
        xaxis_title=spec.time.title,
        yaxis_title=spec.entity_label,
        template="plotly_white",
        legend_title_text=spec.event_label,
    )
    return figure
