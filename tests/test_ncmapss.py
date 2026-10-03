"""N-CMAPSS loader regressions with a tiny synthetic HDF5 fixture."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np
import polars as pl

from pdmdata.datasets.ncmapss import inventory, load
from pdmdata.datasets.ncmapss import loader as ncmapss_loader


def _write_fixture(path: Path) -> None:
    a_dev = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 1, 1],
            [2, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    a_test = np.array([[3, 1, 1, 1]], dtype=np.float64)
    w = np.ones((a_dev.shape[0], 4), dtype=np.float64)
    w_test = np.ones((1, 4), dtype=np.float64)
    xs = np.arange(a_dev.shape[0] * 14, dtype=np.float64).reshape(
        a_dev.shape[0], 14
    )
    xs_test = np.zeros((1, 14), dtype=np.float64)
    y_dev = np.array([[2], [1], [0]], dtype=np.int64)
    y_test = np.array([[5]], dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("A_dev", data=a_dev)
        handle.create_dataset("A_test", data=a_test)
        handle.create_dataset("W_dev", data=w)
        handle.create_dataset("W_test", data=w_test)
        handle.create_dataset("X_s_dev", data=xs)
        handle.create_dataset("X_s_test", data=xs_test)
        handle.create_dataset("X_v_dev", data=xs)
        handle.create_dataset("X_v_test", data=xs_test)
        handle.create_dataset("T_dev", data=np.zeros((3, 10)))
        handle.create_dataset("T_test", data=np.zeros((1, 10)))
        handle.create_dataset("Y_dev", data=y_dev)
        handle.create_dataset("Y_test", data=y_test)
        handle.create_dataset(
            "A_var",
            data=np.array([b"unit", b"cycle", b"Fc", b"hs"]),
        )
        handle.create_dataset(
            "W_var",
            data=np.array([b"alt", b"Mach", b"TRA", b"T2"]),
        )
        handle.create_dataset(
            "X_s_var",
            data=np.array([f"s{i}".encode() for i in range(14)]),
        )
        handle.create_dataset(
            "X_v_var",
            data=np.array([f"v{i}".encode() for i in range(14)]),
        )
        handle.create_dataset(
            "T_var",
            data=np.array([f"t{i}".encode() for i in range(10)]),
        )


class NcmapssTest(TestCase):
    def test_load_inventory_and_unit_filter(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "N-CMAPSS_DS01-005.h5"
            _write_fixture(path)
            with patch.object(ncmapss_loader, "_path", return_value=path):
                frame = load("ds01", "dev", with_rul=True)
                self.assertEqual(frame.height, 3)
                self.assertIn("RUL", frame.columns)
                self.assertEqual(frame["RUL"].to_list(), [2, 1, 0])
                unit = load("ds01", "train", unit=1)
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
