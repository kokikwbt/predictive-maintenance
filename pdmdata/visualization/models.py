"""Typed configuration models for reusable Plotly visualizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AxisSpec:
    """A data column and its human-readable axis metadata."""

    column: str
    label: str
    unit: Optional[str] = None

    @property
    def title(self) -> str:
        """Return an axis title including its unit when available."""
        if self.unit:
            return "{} [{}]".format(self.label, self.unit)
        return self.label


@dataclass(frozen=True)
class TimeSeriesSpec:
    """Configuration for one or more time-series traces."""

    title: str
    x: AxisSpec
    y: Tuple[AxisSpec, ...]
    entity_column: Optional[str] = None
    max_points: int = 10_000


@dataclass(frozen=True)
class DistributionSpec:
    """Configuration for a grouped feature distribution."""

    title: str
    feature: AxisSpec
    group_column: str
    group_label: str
    bins: int = 60


@dataclass(frozen=True)
class EventTimelineSpec:
    """Configuration for a categorical event timeline."""

    title: str
    time: AxisSpec
    entity_column: str
    entity_label: str
    event_column: str
    event_label: str
    max_points: int = 10_000
