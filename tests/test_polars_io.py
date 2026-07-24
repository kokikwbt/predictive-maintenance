import tempfile
import unittest
from pathlib import Path

import polars as pl

import datasets
from datasets.io import read_whitespace
from datasets import loaders


class PolarsIoTest(unittest.TestCase):
    def test_whitespace_reader_returns_polars_dataframe(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "matrix.txt"
            source.write_text("1  2  3  \n4  5  6  \n", encoding="utf-8")

            frame = read_whitespace(source, ["a", "b", "c"])

            self.assertIsInstance(frame, pl.DataFrame)
            self.assertEqual(frame.shape, (2, 3))
            self.assertEqual(frame.columns, ["a", "b", "c"])

    def test_unified_loader_rejects_unknown_dataset(self):
        with self.assertRaisesRegex(KeyError, "Unknown dataset"):
            datasets.load("unknown")

    def test_ims_loader_names_vibration_channels(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "2003.10.22.12.06.24"
            source.write_text("0.1\t0.2\n0.3\t0.4\n", encoding="utf-8")
            with unittest.mock.patch.object(
                loaders, "find_raw_file", return_value=source
            ):
                frame = loaders._ims(source.name)

        self.assertEqual(frame.columns, ["sample", "channel_1", "channel_2"])
        self.assertEqual(frame.shape, (2, 3))

    def test_oyicd_loader_adds_filename_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "01-04T184148_000_mode1.csv"
            source.write_text(
                "timestamp,pCut::Motor_Torque\n0.0,1.5\n",
                encoding="utf-8",
            )
            with unittest.mock.patch.object(
                loaders, "find_raw_file", return_value=source
            ):
                frame = loaders._oyicd(source.name)

        self.assertEqual(frame["month"].to_list(), [1])
        self.assertEqual(frame["mode"].to_list(), [1])
        self.assertEqual(frame["filename"].to_list(), [source.name])

    def test_legacy_dataset_module_api_is_not_exported(self):
        self.assertNotIn("cmapss", datasets.__all__)
        self.assertNotIn("ufd", datasets.__all__)
        self.assertNotIn("__getattr__", vars(datasets))


if __name__ == "__main__":
    unittest.main()
