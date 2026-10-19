"""N-CMAPSS loader regressions with a tiny synthetic HDF5 fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl
from support_ncmapss import write_tiny_fixture

from pdmdata.datasets.ncmapss import inventory, load, loader as ncmapss_loader


class NcmapssTest(TestCase):
    def test_load_inventory_and_unit_filter(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "N-CMAPSS_DS01-005.h5"
            write_tiny_fixture(path)
            with patch.object(ncmapss_loader, "_path", return_value=path):
                frame = load("ds01", "dev", with_rul=True)
                self.assertEqual(frame.height, 3)
                self.assertIn("RUL", frame.columns)
                self.assertEqual(frame["RUL"].to_list(), [2, 1, 0])
                unit = load("ds01", "dev", unit=1)
                self.assertEqual(unit.height, 2)
                self.assertEqual(
                    unit["unit_number"].unique().to_list(),
                    [1],
                )
                summary = inventory("ds01")
                self.assertEqual(summary.height, 2)
                self.assertEqual(
                    summary.filter(pl.col("split") == "dev")["units"][0],
                    2,
                )
                with self.assertRaises(ValueError):
                    load("ds01", "dev", unit=99)
