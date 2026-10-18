"""ALPI source fidelity, selectors, and pinned download behavior."""

import importlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import polars as pl

import pdmdata
from pdmdata.datasets.alpi import inventory, verify


CSV = (
    "timestamp,alarm,serial\n"
    "2020-01-02 10:00:00.123,0,4\n"
    "2020-01-02 10:00:00.123,0,4\n"
    "2020-01-01 09:00:00.000,153,0\n"
    "2020-01-02 10:01:00.456,12,4\n"
)


class AlpiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "alpi/extracted/alarms_log_data/raw/alarms.csv"
        self.path.parent.mkdir(parents=True)
        self.path.write_text(CSV)
        self.settings = patch(
            "pdmdata.io.paths.get_settings", return_value=pdmdata.Settings(self.root)
        )
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def test_ids_precision_duplicates_and_source_order_are_preserved(self):
        scan = pdmdata.load("alpi")
        self.assertIsInstance(scan, pl.LazyFrame)
        frame = scan.collect()
        self.assertEqual(frame.columns, ["source_row", "timestamp", "alarm", "serial"])
        self.assertEqual(frame["alarm"].to_list(), [0, 0, 153, 12])
        self.assertEqual(frame["source_row"].to_list(), [0, 1, 2, 3])
        self.assertEqual(frame.schema["timestamp"], pl.Datetime("us"))
        self.assertEqual(frame["timestamp"][0].microsecond, 123000)
        result = verify()
        self.assertEqual(result["duplicates"], 1)
        self.assertEqual(result["adjacent_ties_per_machine"], 1)
        self.assertEqual(result["out_of_order_per_machine"], 0)
        self.assertEqual(inventory()["events"].to_list(), [1, 3])

    def test_filters_are_half_open_and_keep_original_row_numbers(self):
        frame = pdmdata.load(
            "alpi", serial=[4], start="2020-01-02T10:00:00.123",
            end=datetime(2020, 1, 2, 10, 1, 0, 456000), lazy=False,
        )
        self.assertEqual(frame["source_row"].to_list(), [0, 1])
        selected = pdmdata.load("alpi", alarm=12, columns=["source_row", "alarm"], lazy=False)
        self.assertEqual(selected.rows(), [(3, 12)])
        self.assertEqual(pdmdata.load("alpi", serial=99).collect().height, 0)

    def test_invalid_options_and_schema_do_not_silently_reinterpret(self):
        for options in [dict(serial=True), dict(serial=[]), dict(serial="4"),
                        dict(alarm=-1), dict(start="bad time"),
                        dict(start="2020-01-01T00:00:00+00:00"),
                        dict(start="2020-01-02", end="2020-01-01"),
                        dict(columns="alarm"), dict(columns=[]),
                        dict(columns=["fault"]), dict(columns=["alarm", "alarm"]),
                        dict(lazy=1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                pdmdata.load("alpi", **options)
        self.path.write_text("timestamp,alarm,serial,fault\n2020-01-01 00:00:00.000,1,1,1\n")
        with self.assertRaisesRegex(ValueError, "exactly timestamp"):
            pdmdata.load("alpi")

    def test_verify_reports_out_of_order_without_sorting_and_rejects_nulls(self):
        self.path.write_text(CSV + "2020-01-01 01:00:00.000,12,4\n")
        self.assertEqual(verify()["out_of_order_per_machine"], 1)
        self.assertEqual(pdmdata.load("alpi", lazy=False)["source_row"].to_list(), list(range(5)))
        self.path.write_text("timestamp,alarm,serial\n2020-01-01 00:00:00.000,,1\n")
        with self.assertRaisesRegex(ValueError, "non-null"):
            verify()

    def test_download_is_pinned_and_bad_checksum_prevents_extraction(self):
        info = pdmdata.load_metadata("alpi")
        self.assertEqual(info["download"]["checksum"],
                         "sha256:ecc199cbd5ed461b654b92d546b354954d815afc072d1c59e3dfddf9b1a3f95d")
        self.assertIn("alpi", pdmdata.supported_downloads())
        self.assertNotIn("alpi", pdmdata.bulk_downloads())
        self.assertIn("file_downloaded", pdmdata.download_command("alpi", self.root))
        module = importlib.import_module("pdmdata.io.download")
        with patch.object(module, "_download_file", side_effect=lambda url, target: target.write_bytes(b"not zip")):
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                pdmdata.download("alpi", self.root, overwrite=True)
        self.assertFalse((self.root / "alpi/manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
