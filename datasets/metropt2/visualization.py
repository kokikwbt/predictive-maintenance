"""MetroPT2 visualization configuration."""

from datasets.visualization.models import AxisSpec, TimeSeriesSpec
from datasets.visualization.plots import plot_large_time_series


SENSOR_SPEC = TimeSeriesSpec(
    title="MetroPT2 air-production-unit signals",
    x=AxisSpec("timestamp", "Timestamp"),
    y=(
        AxisSpec("TP2", "Compressor pressure", "bar"),
        AxisSpec("TP3", "Pneumatic-panel pressure", "bar"),
        AxisSpec("Oil_temperature", "Oil temperature", "°C"),
        AxisSpec("Motor_current", "Motor current", "A"),
    ),
    max_points=10_000,
)


def sensor_signals(frame, **options):
    """Plot a bounded set of the principal MetroPT2 analogue signals."""
    return plot_large_time_series(frame, SENSOR_SPEC, **options)


PLOTS = {"sensor_signals": sensor_signals}
