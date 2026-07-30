"""C-MAPSS visualization configuration."""

from datasets.visualization.models import AxisSpec, TimeSeriesSpec


PLOTS = {
    "sensor_trajectory": TimeSeriesSpec(
        title="C-MAPSS sensor trajectory",
        x=AxisSpec("cycle", "Operating cycle"),
        y=(
            AxisSpec("sensor_2", "Sensor 2"),
            AxisSpec("sensor_7", "Sensor 7"),
            AxisSpec("sensor_11", "Sensor 11"),
            AxisSpec("sensor_12", "Sensor 12"),
        ),
        entity_column="unit_number",
    )
}
