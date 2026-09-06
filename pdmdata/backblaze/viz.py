"""Backblaze Drive Stats visualization helpers."""

from __future__ import annotations

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


def plot_waveforms(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    serial_number: str,
    columns: list[str],
):
    """Plot full-resolution SMART histories for one drive on Matplotlib axes."""
    import matplotlib.pyplot as plt

    if not serial_number or not columns:
        raise ValueError("Select a serial_number and at least one SMART column")
    schema = frame.collect_schema() if isinstance(frame, pl.LazyFrame) else frame.schema
    for column in columns:
        if column not in schema or not column.startswith("smart_") or not schema[column].is_numeric():
            raise ValueError(f"Unknown or non-numeric SMART column: {column}")
    if not {"date", "serial_number", "failure"}.issubset(schema):
        raise ValueError("Waveforms require date, serial_number, and failure columns")
    selected = frame.filter(pl.col("serial_number") == serial_number).select(
        list(dict.fromkeys(["date", "failure", *columns]))
    ).sort("date")
    if isinstance(selected, pl.LazyFrame):
        selected = selected.collect()
    if selected.is_empty():
        raise ValueError(f"No observations for drive {serial_number!r}")
    if selected["date"].n_unique() != selected.height:
        raise ValueError("A drive history must contain at most one row per date")
    figure, axes = plt.subplots(len(columns), 1, sharex=True, squeeze=False,
                                figsize=(12, 2.5 * len(columns)), layout="constrained")
    failure_dates = selected.filter(pl.col("failure") == 1)["date"].to_list()
    for column, axis in zip(columns, axes[:, 0]):
        axis.plot(selected["date"].to_list(), selected[column].to_numpy(), linewidth=0.8)
        for index, day in enumerate(failure_dates):
            axis.axvline(day, color="tab:red", linestyle="--",
                        label="Reported failure" if index == 0 else None)
        axis.set_ylabel(column)
        axis.grid(alpha=0.2)
        if failure_dates:
            axis.legend()
    axes[-1, 0].set_xlabel("Date")
    figure.suptitle(f"Backblaze SMART history: {serial_number}")
    return figure


def save_samples(output_dir=None):
    """Save three compact, reproducible real-data examples from local 2025 Q1.

    These deliberately selected examples illustrate histories, not fleet-level
    performance. No downloads, raw-data exports, or synthetic values are used.
    """
    from pathlib import Path
    import json
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from .loader import load

    output = Path(output_dir) if output_dir is not None else Path(__file__).parent / "assets"
    examples = [
        ("ssd-january", "2207E60CC65A", ["smart_5_raw", "smart_9_raw", "smart_194_raw"]),
        ("hgst-failure-january", "8DHBLDBH", ["smart_5_raw", "smart_197_raw", "smart_194_raw"]),
        ("seagate-failure-january", "ZHZ3MAAX", ["smart_5_raw", "smart_197_raw", "smart_194_raw"]),
    ]
    signals = list(dict.fromkeys(c for _, _, columns in examples for c in columns))
    serials = [serial for _, serial, _ in examples]
    # Scan the selected month once for all examples, collecting only 3 drives.
    frame = load(
        variant="2025-q1", start_date="2025-01-01", end_date="2025-01-31",
        columns=["date", "serial_number", "model", "failure", *signals],
    ).filter(pl.col("serial_number").is_in(serials)).collect()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, serial, columns in examples:
        selected = frame.filter(pl.col("serial_number") == serial).sort("date")
        if selected.is_empty():
            raise ValueError(f"Sample drive is missing from local data: {serial}")
        model = selected["model"][0]
        figure = plot_waveforms(selected, serial_number=serial, columns=columns)
        figure.set_size_inches(10, 6)
        figure.suptitle(f"{model} / {serial}\nJanuary 2025 · daily observations", fontsize=12)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=6)
        figure.axes[-1].xaxis.set_major_locator(locator)
        figure.axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        path = output / f"{name}.png"
        figure.savefig(path, dpi=110, pil_kwargs={"optimize": True})
        plt.close(figure)
        records.append({
            "image": path.name, "variant": "2025-q1", "serial_number": serial,
            "model": model, "start_date": str(selected["date"].min()),
            "end_date": str(selected["date"].max()), "observations": selected.height,
            "columns": columns,
            "failure_dates": [str(d) for d in selected.filter(pl.col("failure") == 1)["date"]],
            "bytes": path.stat().st_size,
        })
    manifest = {
        "source": "https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data",
        "selection": "One SSD history and two drives with a reported failure on January 31, 2025. Illustrative selection, not a representative evaluation sample.",
        "processing": "Original daily values; no smoothing, resampling, normalization, or zero imputation. SMART values retain provider-specific semantics.",
        "samples": records,
    }
    (output / "samples.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return records
