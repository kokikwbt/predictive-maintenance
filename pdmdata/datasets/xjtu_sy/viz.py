"""XJTU-SY Plotly specifications for bearing degradation."""

from pdmdata.visualization.models import AxisSpec, TimeSeriesSpec


PLOTS = {
    "rms_trajectory": TimeSeriesSpec(
        title="XJTU-SY bearing RMS trajectory",
        x=AxisSpec("cycle", "Snapshot cycle"),
        y=(
            AxisSpec("rms_horizontal", "Horizontal RMS"),
            AxisSpec("rms_vertical", "Vertical RMS"),
        ),
        max_points=10_000,
    ),
}
