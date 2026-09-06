"""Reusable interactive visualizations for predictive-maintenance datasets."""

from .models import AxisSpec, DistributionSpec, EventTimelineSpec, TimeSeriesSpec
from .plots import (
    plot_distribution,
    plot_event_timeline,
    plot_large_time_series,
    plot_time_series,
)
from .registry import available_plots, visualize


__all__ = [
    "AxisSpec",
    "DistributionSpec",
    "EventTimelineSpec",
    "TimeSeriesSpec",
    "available_plots",
    "plot_distribution",
    "plot_event_timeline",
    "plot_large_time_series",
    "plot_time_series",
    "visualize",
]
