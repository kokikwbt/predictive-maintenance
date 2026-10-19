"""ALPI Plotly specifications for alarm event timelines."""

from pdmdata.visualization.models import AxisSpec, EventTimelineSpec

PLOTS = {
    "alarm_timeline": EventTimelineSpec(
        title="ALPI alarm timeline",
        time=AxisSpec("timestamp", "Event timestamp"),
        entity_column="serial",
        entity_label="Machine serial",
        event_column="alarm",
        event_label="Alarm code",
    ),
}
