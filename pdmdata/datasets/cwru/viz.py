"""Waveform and Welch spectrum views of source-faithful CWRU frames."""

import numpy as np
import plotly.graph_objects as go
import polars as pl
from scipy.signal import welch


def _select(frame, recording=None, channel=None):
    if isinstance(frame, pl.LazyFrame):
        frame = frame.collect()
    required = {"recording", "channel", "sample", "amplitude"}
    if not required <= set(frame.columns):
        raise ValueError(
            f"Missing CWRU plotting columns: {sorted(required - set(frame.columns))}"
        )
    if recording is not None:
        frame = frame.filter(pl.col("recording") == recording)
    if channel is not None:
        frame = frame.filter(pl.col("channel") == channel)
    if frame.is_empty():
        raise ValueError("No CWRU samples match the plot selection")
    return frame


def waveform(frame, *, recording=None, channel=None, max_points=10000):
    """Plot separate recording/channel traces with deterministic point selection."""
    if type(max_points) is not int or max_points < 2:
        raise ValueError("max_points must be an integer >= 2")
    frame = _select(frame, recording, channel)
    figure = go.Figure()
    for group in frame.partition_by(
        ["recording", "channel"], maintain_order=True
    ):
        group = group.sort("sample")
        positions = np.linspace(
            0, group.height - 1, min(max_points, group.height), dtype=int
        )
        figure.add_trace(
            go.Scattergl(
                x=group["sample"].to_numpy()[positions],
                y=group["amplitude"].to_numpy()[positions],
                mode="lines",
                name=f"{group['recording'][0]} / {group['channel'][0]}",
            )
        )
    figure.update_layout(
        title="CWRU vibration waveforms",
        xaxis_title="Original sample index",
        yaxis_title="Source amplitude",
    )
    return figure


def spectrum(
    frame, *, recording=None, channel=None, sample_rate_hz=None, nperseg=2048
):
    """Compute Welch PSD for one uniformly spaced recording/channel series.

    Sliced samples use the original sampling rate divided by sample-index step.
    No spectrum is computed from unspecified rates without an explicit rate.
    """
    frame = _select(frame, recording, channel).sort("sample")
    if frame.select(pl.struct("recording", "channel").n_unique()).item() != 1:
        raise ValueError("Select one recording and one channel for a spectrum")
    if type(nperseg) is not int or nperseg < 2 or frame.height < 2:
        raise ValueError(
            "A spectrum needs >= 2 samples and integer nperseg >= 2"
        )
    if sample_rate_hz is None:
        if (
            "sample_rate_hz" not in frame.columns
            or frame["sample_rate_hz"].null_count()
        ):
            raise ValueError(
                "Sampling rate is unspecified; provide sample_rate_hz"
            )
        if frame["sample_rate_hz"].n_unique() != 1:
            raise ValueError("Inconsistent sampling rates")
        sample_rate_hz = frame["sample_rate_hz"][0]
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, (int, float))
        or not np.isfinite(sample_rate_hz)
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be a positive finite number")
    increments = np.diff(frame["sample"].cast(pl.Int64).to_numpy())
    if increments[0] <= 0 or not np.all(increments == increments[0]):
        raise ValueError(
            "Spectrum requires unique uniformly spaced sample indices"
        )
    frequencies, density = welch(
        frame["amplitude"].to_numpy(),
        fs=sample_rate_hz / increments[0],
        nperseg=min(nperseg, frame.height),
    )
    figure = go.Figure(go.Scatter(x=frequencies, y=density, mode="lines"))
    figure.update_layout(
        title="CWRU Welch power spectral density",
        xaxis_title="Frequency [Hz]",
        yaxis_title="Power spectral density [source amplitude²/Hz]",
    )
    return figure


PLOTS = {"waveform": waveform, "spectrum": spectrum}
