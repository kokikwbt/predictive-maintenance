"""Dataset-specific visualization registry and high-level dispatcher."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import plotly.graph_objects as go
import polars as pl

from .models import DistributionSpec, EventTimelineSpec, TimeSeriesSpec
from .plots import plot_distribution, plot_event_timeline, plot_time_series

_PLOTS: dict[str, dict[str, object]] | None = None


def _plots() -> dict[str, dict[str, object]]:
    """Load dataset plot specs on first use (avoid import-time side effects)."""
    global _PLOTS
    if _PLOTS is not None:
        return _PLOTS
    from pdmdata.datasets.alpi.viz import PLOTS as ALPI_PLOTS
    from pdmdata.datasets.backblaze.viz import PLOTS as BACKBLAZE_PLOTS
    from pdmdata.datasets.care.viz import PLOTS as CARE_PLOTS
    from pdmdata.datasets.cmapss.viz import PLOTS as CMAPSS_PLOTS
    from pdmdata.datasets.cwru.viz import PLOTS as CWRU_PLOTS
    from pdmdata.datasets.gdd.viz import PLOTS as GDD_PLOTS
    from pdmdata.datasets.gfd.viz import PLOTS as GFD_PLOTS
    from pdmdata.datasets.hydsys.viz import PLOTS as HYDSYS_PLOTS
    from pdmdata.datasets.ims.viz import PLOTS as IMS_PLOTS
    from pdmdata.datasets.mapm.viz import PLOTS as MAPM_PLOTS
    from pdmdata.datasets.metropt2.viz import PLOTS as METROPT2_PLOTS
    from pdmdata.datasets.ncmapss.viz import PLOTS as NCMAPSS_PLOTS
    from pdmdata.datasets.oyicd.viz import PLOTS as OYICD_PLOTS
    from pdmdata.datasets.ppd.viz import PLOTS as PPD_PLOTS
    from pdmdata.datasets.scania_x.viz import PLOTS as SCANIA_X_PLOTS
    from pdmdata.datasets.xjtu_sy.viz import PLOTS as XJTU_SY_PLOTS

    _PLOTS = {
        "alpi": ALPI_PLOTS,
        "backblaze": BACKBLAZE_PLOTS,
        "care": CARE_PLOTS,
        "cmapss": CMAPSS_PLOTS,
        "cwru": CWRU_PLOTS,
        "gdd": GDD_PLOTS,
        "gfd": GFD_PLOTS,
        "hydsys": HYDSYS_PLOTS,
        "ims": IMS_PLOTS,
        "mapm": MAPM_PLOTS,
        "metropt2": METROPT2_PLOTS,
        "ncmapss": NCMAPSS_PLOTS,
        "oyicd": OYICD_PLOTS,
        "ppd": PPD_PLOTS,
        "scania_x": SCANIA_X_PLOTS,
        "xjtu_sy": XJTU_SY_PLOTS,
    }
    return _PLOTS


def available_plots(dataset_id: str | None = None) -> dict[str, object]:
    """Return registered plot specifications."""
    plots = _plots()
    if dataset_id is None:
        return {key: sorted(value) for key, value in plots.items()}
    try:
        return dict(plots[dataset_id])
    except KeyError:
        raise KeyError(
            f"No visualizations registered for {dataset_id!r}"
        ) from None


def visualize(
    dataset_id: str,
    plot_id: str,
    frame: pl.DataFrame | pl.LazyFrame,
    **options: Any,
) -> go.Figure:
    """Build a registered dataset visualization."""
    plots = _plots()
    try:
        spec = plots[dataset_id][plot_id]
    except KeyError:
        choices = ", ".join(sorted(plots.get(dataset_id, {}))) or "none"
        raise KeyError(
            f"Unknown visualization {plot_id!r} for {dataset_id!r}; available: {choices}"
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
    raise TypeError(
        f"Unsupported visualization specification: {type(spec)!r}"
    )
