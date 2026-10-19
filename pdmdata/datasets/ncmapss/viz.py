"""N-CMAPSS Plotly specifications for unit trajectories."""

from pdmdata.visualization.models import AxisSpec, TimeSeriesSpec

PLOTS = {
    "sensor_trajectory": TimeSeriesSpec(
        title="N-CMAPSS sensor trajectory",
        x=AxisSpec("cycle", "Operating cycle"),
        y=(
            AxisSpec("T24", "T24"),
            AxisSpec("T30", "T30"),
            AxisSpec("T50", "T50"),
            AxisSpec("P30", "P30"),
        ),
        entity_column="unit_number",
        max_points=10_000,
    ),
}
