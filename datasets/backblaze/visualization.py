"""Backblaze Drive Stats visualization helpers."""

import plotly.graph_objects as go
import polars as pl


def daily_failure_rate(frame: pl.DataFrame | pl.LazyFrame) -> go.Figure:
    """Aggregate raw drive snapshots into an interactive daily failure-rate plot."""
    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    daily = (
        lazy.select("date", "failure")
        .with_columns(
            pl.col("date").cast(pl.String).str.to_date(strict=False)
        )
        .group_by("date")
        .agg(
            pl.len().alias("drive_count"),
            pl.col("failure").sum().alias("failure_count"),
        )
        .with_columns(
            (100 * pl.col("failure_count") / pl.col("drive_count"))
            .alias("failure_rate_percent")
        )
        .sort("date")
        .collect()
    )
    figure = go.Figure(
        go.Scatter(
            x=daily["date"].to_list(),
            y=daily["failure_rate_percent"].to_list(),
            mode="lines+markers",
            customdata=daily.select("failure_count", "drive_count").to_numpy(),
            hovertemplate=(
                "%{x}<br>Failure rate: %{y:.4f}%"
                "<br>Failures: %{customdata[0]}"
                "<br>Drives: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="Backblaze daily drive failure rate",
        xaxis_title="Date",
        yaxis_title="Failed drives [%]",
        hovermode="x unified",
        template="plotly_white",
    )
    return figure


PLOTS = {"daily_failure_rate": daily_failure_rate}
