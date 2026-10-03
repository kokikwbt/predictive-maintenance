import tempfile
import unittest
from pathlib import Path

import polars as pl

import pdmdata
from pdmdata.io import read_whitespace
from pdmdata.datasets.gfd import loader as gfd_loader
from pdmdata.datasets.oyicd import loader as oyicd_loader
from pdmdata.datasets.ppd import loader as ppd_loader
from pdmdata.datasets.metropt2 import loader as metropt2_loader
from pdmdata.datasets.backblaze import loader as backblaze_loader


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
            pdmdata.load("unknown")

    def test_gfd_loader_reads_tab_delimited_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "h30hz0.txt"
            source.write_text(
                "\n1.0\t2.0\t3.0\t4.0\t\n5.0\t6.0\t7.0\t8.0\t\n\n",
                encoding="utf-8",
            )
            with unittest.mock.patch.object(
                gfd_loader, "find_raw_file", return_value=source
            ):
                frame = gfd_loader.load()

        self.assertEqual(
            frame.columns,
            ["sensor_1", "sensor_2", "sensor_3", "sensor_4", "condition", "load"],
        )
        self.assertEqual(frame.shape, (2, 6))

    def test_gfd_loader_rejects_incomplete_or_nonfinite_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "h30hz0.txt"
            for row in ["5\t\t7\t8\t\n", "5\tNaN\t7\t8\t\n", "5\tinf\t7\t8\t\n"]:
                with self.subTest(row=row):
                    source.write_text("1\t2\t3\t4\t\n" + row)
                    with unittest.mock.patch.object(gfd_loader, "find_raw_file", return_value=source):
                        with self.assertRaisesRegex(ValueError, "Missing or non-finite"):
                            gfd_loader.load()

    def test_oyicd_loader_adds_filename_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "01-04T184148_000_mode1.csv"
            source.write_text(
                ",".join(["timestamp", *oyicd_loader.SENSOR_COLUMNS]) + "\n"
                + "0.0," + ",".join(["1.5"] * 8) + "\n"
                + "0.004," + ",".join(["2.5"] * 8) + "\n",
                encoding="utf-8",
            )
            with unittest.mock.patch.object(
                oyicd_loader, "_find_recording", return_value=source
            ):
                frame = oyicd_loader.load(source.name)

        self.assertEqual(frame["month"].to_list(), [1, 1])
        self.assertEqual(frame["mode"].to_list(), [1, 1])
        self.assertEqual(frame["filename"].to_list(), [source.name, source.name])

    def test_metropt2_loader_is_lazy_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "MetroPT2.csv"
            source.write_text("timestamp,pressure\n2023-01-01,1.5\n", encoding="utf-8")
            with unittest.mock.patch.object(
                metropt2_loader, "find_raw_file", return_value=source
            ):
                frame = metropt2_loader.load()
                shape = frame.collect().shape

        self.assertIsInstance(frame, pl.LazyFrame)
        self.assertEqual(shape, (1, 2))

    def test_ppd_loader_preserves_missing_rows_and_rejects_bad_time_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "C7-1.csv"
            header = ",".join(["Timestamp", *ppd_loader.SENSOR_COLUMNS]) + "\n"
            contents = header + "0," + ",".join(["1.5"] * 25) + "\n"
            contents += "1," + ",".join([""] * 25) + "\n"
            contents += "2," + ",".join(["2.5"] * 25) + "\n"
            source.write_text(contents)
            with unittest.mock.patch.object(ppd_loader, "find_raw_file", return_value=source):
                frame = ppd_loader.load_file("C7-1")
                self.assertEqual(frame["Timestamp"].to_list(), [0, 1, 2])
                self.assertEqual(frame["L_1"].to_list(), [1.5, None, 2.5])
                self.assertEqual(frame["source_file"].unique().to_list(), ["C7-1.csv"])
                self.assertEqual(frame.width, 28)
                source.write_text(contents.replace("\n2,", "\n0,"))
                with self.assertRaisesRegex(ValueError, "increase"):
                    ppd_loader.load_file("C7-1")

    def test_ppd_sequence_loading_joins_parts_and_inventory_has_eight_sequences(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "ppd" / "extracted"
            directory.mkdir(parents=True)
            header = ",".join(["Timestamp", *ppd_loader.SENSOR_COLUMNS]) + "\n"
            for name in sorted(ppd_loader.PPD_FILES, reverse=True):
                value = "2.5" if name.endswith("-2") else "1.5"
                contents = header + "0," + ",".join([value] * 25) + "\n"
                contents += "1," + ",".join([""] * 25) + "\n"
                (directory / f"{name}.csv").write_text(contents)
            with unittest.mock.patch(
                "pdmdata.io.paths.get_settings",
                return_value=pdmdata.Settings(data_root=root),
            ):
                for sequence_id in [7, 13]:
                    with self.subTest(sequence_id=sequence_id):
                        frame = ppd_loader.load(sequence_id)
                        self.assertEqual(frame["sample"].to_list(), [0, 1, 2, 3])
                        self.assertEqual(frame["Timestamp"].to_list(), [0, 1, 0, 1])
                        self.assertEqual(frame["L_1"].to_list(), [1.5, None, 2.5, None])
                        self.assertEqual(frame["source_file"].to_list(),
                                         [f"C{sequence_id}-1.csv"] * 2 + [f"C{sequence_id}-2.csv"] * 2)
                        self.assertEqual(frame["sequence_id"].unique().to_list(), [sequence_id])
                        self.assertTrue(frame.equals(ppd_loader.load(f"C{sequence_id}")))
                self.assertEqual(ppd_loader.load(8).height, 2)
                summary = ppd_loader.inventory()
                self.assertEqual(summary["sequence_id"].to_list(), [7, 8, 9, 11, 13, 14, 15, 16])
                self.assertEqual(summary["parts"].sum(), 10)
                self.assertEqual(summary["samples"].sum(), 20)
                with self.assertRaisesRegex(ValueError, "sequence_id"):
                    ppd_loader.load("C7-1")
                (directory / "C7-2.csv").unlink()
                with self.assertRaises(FileNotFoundError):
                    ppd_loader.load(7)

    def test_oyicd_duplicate_copies_are_verified_and_counted_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "oyicd" / "extracted"
            duplicate = directory / "oneyeardata"
            duplicate.mkdir(parents=True)
            name = "01-04T184148_000_mode1.csv"
            contents = ",".join(["timestamp", *oyicd_loader.SENSOR_COLUMNS]) + "\n"
            contents += "0.0," + ",".join(["1.5"] * 8) + "\n"
            contents += "0.004," + ",".join(["2.5"] * 8) + "\n"
            (directory / name).write_text(contents)
            (duplicate / name).write_text(contents)
            with unittest.mock.patch.object(
                oyicd_loader, "get_settings", return_value=pdmdata.Settings(data_root=root)
            ):
                files = oyicd_loader.inventory(month=1, mode=1)
                self.assertEqual(files["copies"].to_list(), [2])
                self.assertEqual(files["samples"].to_list(), [2])
                self.assertTrue(oyicd_loader.inventory(mode=8).is_empty())
                self.assertEqual(oyicd_loader.load(name).height, 2)
                (duplicate / name).write_text(contents.replace("2.5", "3.5"))
                with self.assertRaisesRegex(ValueError, "Conflicting copies"):
                    oyicd_loader.load(name)

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
                backblaze_loader, "get_settings", return_value=pdmdata.Settings(data_root=root)
            ):
                frame = backblaze_loader.load()
                shape = frame.collect().shape

        self.assertIsInstance(frame, pl.LazyFrame)
        self.assertEqual(shape, (1, 3))

if __name__ == "__main__":
    unittest.main()
