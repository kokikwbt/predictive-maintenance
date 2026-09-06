from datetime import datetime, timedelta
import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from pdmdata.care.viz import plot_waveforms


class CareWaveformTest(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_full_recording_and_numeric_signal_selection(self):
        count = 10002
        times = [datetime(2022, 1, 1) + timedelta(minutes=10 * i) for i in range(count)]
        frame = pl.DataFrame({
            "time_stamp": times, "asset_id": [1] * count,
            "sensor_0_avg": range(count), "power_1_avg": range(count),
            "train_test": ["train"] * 10000 + ["prediction"] * 2,
        }).lazy()
        fig = plot_waveforms(frame)
        self.assertEqual(len(fig.axes), 2)
        for axis in fig.axes:
            self.assertEqual(len(axis.lines[0].get_xdata()), count)
            self.assertEqual(axis.lines[0].get_xdata()[-1], times[-1])
            self.assertEqual(len(axis.patches), 1)
        self.assertEqual(fig.axes[0].get_ylabel(), "sensor_0_avg")

    def test_empty_selection_and_non_numeric_columns_are_rejected(self):
        frame = pl.DataFrame({"time_stamp": [datetime(2022, 1, 1)], "label": ["normal"], "sensor_0_avg": [1.]})
        for columns in ([], ["label"], ["missing"]):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                plot_waveforms(frame, columns=columns)
        with self.assertRaisesRegex(ValueError, "No CARE observations"):
            plot_waveforms(frame.head(0))
