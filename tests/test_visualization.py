import unittest

import plotly.graph_objects as go
import polars as pl

import pdmdata
from pdmdata.visualization import AxisSpec, TimeSeriesSpec, plot_time_series


class VisualizationTest(unittest.TestCase):
    def test_registered_time_series_uses_dataset_labels(self):
        frame = pl.DataFrame(
            {
                "unit_number": [1, 1, 1],
                "cycle": [1, 2, 3],
                "sensor_2": [0.1, 0.2, 0.3],
                "sensor_7": [1.1, 1.2, 1.3],
                "sensor_11": [2.1, 2.2, 2.3],
                "sensor_12": [3.1, 3.2, 3.3],
            }
        )

        figure = pdmdata.visualize(
            "cmapss", "sensor_trajectory", frame, entity=1
        )

        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(len(figure.data), 4)
        self.assertEqual(figure.layout.xaxis.title.text, "Operating cycle")
        self.assertIn("unit_number 1", figure.layout.title.text)

    def test_shared_time_series_downsamples_large_frames(self):
        frame = pl.DataFrame({"time": range(100), "value": range(100)})
        spec = TimeSeriesSpec(
            title="Example",
            x=AxisSpec("time", "Time", "s"),
            y=(AxisSpec("value", "Value"),),
            max_points=10,
        )

        figure = plot_time_series(frame, spec)

        self.assertLessEqual(len(figure.data[0].x), 10)
        self.assertEqual(figure.layout.xaxis.title.text, "Time [s]")

    def test_missing_columns_have_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "sensor_12"):
            pdmdata.visualize(
                "cmapss",
                "sensor_trajectory",
                pl.DataFrame(
                    {
                        "unit_number": [1],
                        "cycle": [1],
                        "sensor_2": [0.1],
                        "sensor_7": [0.1],
                        "sensor_11": [0.1],
                    }
                ),
                entity=1,
            )

    def test_metropt2_visualization_accepts_lazy_data(self):
        frame = pl.DataFrame(
            {
                "timestamp": [1, 2],
                "TP2": [7.0, 7.1],
                "TP3": [8.0, 8.1],
                "Oil_temperature": [45.0, 45.2],
                "Motor_current": [2.0, 2.2],
            }
        ).lazy()

        figure = pdmdata.visualize("metropt2", "sensor_signals", frame)

        self.assertEqual(len(figure.data), 4)
        self.assertEqual(figure.layout.xaxis.title.text, "Timestamp")

    def test_care_visualization_supports_explicit_schema(self):
        frame = pl.DataFrame(
            {
                "recorded_at": ["2025-01-01", "2025-01-02"],
                "power_mean": [100.0, 105.0],
                "temperature_mean": [20.0, 21.0],
            }
        ).lazy()

        figure = pdmdata.visualize(
            "care",
            "scada_signals",
            frame,
            time_column="recorded_at",
            columns=["power_mean", "temperature_mean"],
        )

        self.assertEqual(len(figure.data), 2)

        native = pl.DataFrame({
            "time_stamp": ["2025-01-01", "2025-01-02"],
            "asset_id": [1, 1], "id": [0, 1], "status_type_id": [0, 0],
            "sensor_0_avg": [20., 21.], "wind_speed_3_avg": [8., 9.],
        })
        for frame in (native, native.lazy()):
            figure = pdmdata.visualize("care", "scada_signals", frame)
            self.assertEqual([list(trace.y) for trace in figure.data], [[20., 21.], [8., 9.]])

    def test_backblaze_visualization_aggregates_daily_failures(self):
        frame = pl.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-01", "2025-01-02"],
                "failure": [0, 1, 0],
            }
        ).lazy()

        figure = pdmdata.visualize(
            "backblaze", "daily_failure_rate", frame
        )

        self.assertEqual(list(figure.data[0].y), [50.0, 0.0])
        self.assertEqual(figure.layout.yaxis.title.text, "Failed drives [%]")


if __name__ == "__main__":
    unittest.main()
