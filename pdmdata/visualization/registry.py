"""Dataset-specific visualization registry and high-level dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict, Optional

import plotly.graph_objects as go
import polars as pl

from pdmdata.cmapss.viz import PLOTS as CMAPSS_PLOTS
from pdmdata.gfd.viz import PLOTS as GFD_PLOTS
from pdmdata.mapm.viz import PLOTS as MAPM_PLOTS
from pdmdata.metropt2.viz import PLOTS as METROPT2_PLOTS
from pdmdata.care.viz import PLOTS as CARE_PLOTS
from pdmdata.backblaze.viz import PLOTS as BACKBLAZE_PLOTS

from .models import DistributionSpec, EventTimelineSpec, TimeSeriesSpec
from .plots import plot_distribution, plot_event_timeline, plot_time_series


PLOTS: Dict[str, Dict[str, object]] = {
    "cmapss": CMAPSS_PLOTS,
    "backblaze": BACKBLAZE_PLOTS,
    "care": CARE_PLOTS,
    "gfd": GFD_PLOTS,
    "mapm": MAPM_PLOTS,
    "metropt2": METROPT2_PLOTS,
}


def available_plots(dataset_id: Optional[str] = None) -> Dict[str, object]:
    """Return registered plot specifications."""
    if dataset_id is None:
        return {key: sorted(value) for key, value in PLOTS.items()}
    try:
        return dict(PLOTS[dataset_id])
    except KeyError:
        raise KeyError("No visualizations registered for {!r}".format(dataset_id)) from None


def visualize(
    dataset_id: str,
    plot_id: str,
    frame: pl.DataFrame | pl.LazyFrame,
    **options: Any,
) -> go.Figure:
    """Build a registered dataset visualization."""
    try:
        spec = PLOTS[dataset_id][plot_id]
    except KeyError:
        choices = ", ".join(sorted(PLOTS.get(dataset_id, {}))) or "none"
        raise KeyError(
            "Unknown visualization {!r} for {!r}; available: {}".format(
                plot_id, dataset_id, choices
            )
        ) from None
    if isinstance(spec, TimeSeriesSpec):
        if isinstance(frame, pl.LazyFrame):
            from .plots import plot_large_time_series

            return plot_large_time_series(frame, spec, **options)
        return plot_time_series(frame, spec, **options)
    if isinstance(spec, Callable):
        return spec(frame, **options)
    if isinstance(spec, DistributionSpec):
        if options:
            raise TypeError("Distribution plots do not accept options")
        return plot_distribution(frame, spec)
    if isinstance(spec, EventTimelineSpec):
        return plot_event_timeline(frame, spec, **options)
    raise TypeError("Unsupported visualization specification: {!r}".format(type(spec)))
