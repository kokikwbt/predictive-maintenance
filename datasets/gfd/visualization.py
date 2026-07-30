"""GFD visualization configuration."""

from datasets.visualization.models import AxisSpec, DistributionSpec


PLOTS = {
    "condition_distribution": DistributionSpec(
        title="Gearbox vibration by condition",
        feature=AxisSpec("sensor_1", "Vibration sensor 1"),
        group_column="condition",
        group_label="Gear condition",
    )
}
