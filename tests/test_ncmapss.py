"""N-CMAPSS loader regressions with a tiny synthetic HDF5 fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np
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

    def test_unit_filter_reads_only_selected_rows(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "N-CMAPSS_DS01-005.h5"
            write_tiny_fixture(path)
            with patch.object(ncmapss_loader, "_path", return_value=path):
                full = load(
                    "ds01",
                    "dev",
                    include_virtual=True,
                    include_health_modifiers=True,
                )
                unit = load(
                    "ds01",
                    "dev",
                    unit=2,
                    include_virtual=True,
                    include_health_modifiers=True,
                )
            self.assertTrue(
                unit.equals(full.filter(pl.col("unit_number") == 2))
            )

    def test_unit_rows_slice_contiguous_and_index_scattered_blocks(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "units.h5"
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "A", data=np.array([[1], [1], [2], [1]], dtype=float)
                )
            with h5py.File(path, "r") as handle:
                rows = ncmapss_loader._unit_rows(handle["A"], 0, 2)
                self.assertEqual(rows, slice(2, 3))
                rows = ncmapss_loader._unit_rows(handle["A"], 0, 1)
                self.assertEqual(rows.tolist(), [0, 1, 3])
                self.assertIsNone(ncmapss_loader._unit_rows(handle["A"], 0, 9))
