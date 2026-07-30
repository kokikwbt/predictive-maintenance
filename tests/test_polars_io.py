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

    def test_gfd_loader_reads_tab_delimited_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "h30hz0.txt"
            source.write_text(
                "\n1.0\t2.0\t3.0\t4.0\t\n5.0\t6.0\t7.0\t8.0\t\n",
                encoding="utf-8",
            )
            with unittest.mock.patch.object(
                loaders, "find_raw_file", return_value=source
            ):
                frame = loaders._gfd()

        self.assertEqual(
            frame.columns,
            ["sensor_1", "sensor_2", "sensor_3", "sensor_4", "condition", "load"],
        )
        self.assertEqual(frame.shape, (2, 6))

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

    def test_metropt2_loader_is_lazy_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "MetroPT2.csv"
            source.write_text("timestamp,pressure\n2023-01-01,1.5\n", encoding="utf-8")
            with unittest.mock.patch.object(
                loaders, "find_raw_file", return_value=source
            ):
                frame = loaders._metropt2()
                shape = frame.collect().shape

        self.assertIsInstance(frame, pl.LazyFrame)
        self.assertEqual(shape, (1, 2))

    def test_care_loader_reads_one_requested_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "event.csv"
            source.write_text("timestamp,power\n2023-01-01,4.0\n", encoding="utf-8")
            with unittest.mock.patch.object(
                loaders, "find_raw_file", return_value=source
            ):
                frame = loaders._care(source.name, lazy=False)

        self.assertIsInstance(frame, pl.DataFrame)
        self.assertEqual(frame.shape, (1, 2))

    def test_backblaze_loader_scans_a_quarter_lazily(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extracted = root / "backblaze" / "2025-q1" / "extracted"
            extracted.mkdir(parents=True)
            (extracted / "2025-01-01.csv").write_text(
                "date,serial_number,failure\n2025-01-01,A1,0\n",
                encoding="utf-8",
            )
            with unittest.mock.patch.object(
                loaders, "DEFAULT_DATA_ROOT", root
            ):
                frame = loaders._backblaze()
                shape = frame.collect().shape

        self.assertIsInstance(frame, pl.LazyFrame)
        self.assertEqual(shape, (1, 3))

    def test_legacy_dataset_module_api_is_not_exported(self):
        self.assertNotIn("cmapss", datasets.__all__)
        self.assertNotIn("ufd", datasets.__all__)
        self.assertNotIn("__getattr__", vars(datasets))


if __name__ == "__main__":
    unittest.main()
