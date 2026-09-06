"""MAPM visualization configuration."""

from pdmdata.visualization.models import AxisSpec, EventTimelineSpec, TimeSeriesSpec


PLOTS = {
    "telemetry": TimeSeriesSpec(
        title="MAPM machine telemetry",
        x=AxisSpec("datetime", "Timestamp"),
        y=(
            AxisSpec("volt", "Voltage"),
            AxisSpec("rotate", "Rotation"),
            AxisSpec("pressure", "Pressure"),
            AxisSpec("vibration", "Vibration"),
        ),
        entity_column="machineID",
        max_points=5_000,
    ),
    "error_timeline": EventTimelineSpec(
        title="MAPM error timeline",
        time=AxisSpec("datetime", "Timestamp"),
        entity_column="machineID",
        entity_label="Machine ID",
        event_column="errorID",
        event_label="Error type",
    ),
}
