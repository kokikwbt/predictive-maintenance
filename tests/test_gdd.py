"""GDD series selection and state-segment helpers."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

import pdmdata
from pdmdata.datasets.gdd.preprocessing import state_segments

HEADER = (
    "Timestamp,Label,MotorData.ActCurrent,MotorData.ActPosition,"
    "MotorData.ActSpeed,MotorData.IsAcceleration,MotorData.IsForce\n"
)


def _write_series(root: Path, name: str, rows: str) -> None:
    directory = root / "gdd"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(HEADER + rows)


class GddTest(unittest.TestCase):
    def test_series_selection_and_invalid_options(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_series(
                root,
                "Genesis_StateMachineLabel.csv",
                "1.0,0,1,2,3,4,5\n2.0,0,1,2,3,4,5\n3.0,1,1,2,3,4,5\n",
            )
            _write_series(
                root,
                "Genesis_AnomalyLabels.csv",
                "1.0,0,1,2,3,4,5\n2.0,0,1,2,3,4,5\n3.0,1,1,2,3,4,5\n",
            )
            with patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                state = pdmdata.load("gdd", series="state")
                self.assertEqual(state.height, 3)
                self.assertEqual(state["Label"].to_list(), [0, 0, 1])
                anomaly = pdmdata.load("gdd", series="anomaly")
                self.assertEqual(
                    anomaly.drop("Label").to_dicts(),
                    state.drop("Label").to_dicts(),
                )
                with self.assertRaisesRegex(ValueError, "series must be"):
                    pdmdata.load("gdd", series="unknown")

    def test_state_segments_preserve_source_order(self):
        frame = pl.DataFrame({"Label": [0, 0, 1, 1, 1, 0]})
        segments = state_segments(frame)
        self.assertEqual(segments["Label"].to_list(), [0, 1, 0])
        self.assertEqual(segments["start_sample"].to_list(), [0, 2, 5])
        self.assertEqual(segments["end_sample"].to_list(), [2, 5, 6])
        self.assertEqual(segments["samples"].to_list(), [2, 3, 1])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            state_segments(pl.DataFrame({"Label": []}))


if __name__ == "__main__":
    unittest.main()
