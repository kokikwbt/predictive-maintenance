"""XJTU-SY snapshot inventory, RMS features, and derived RUL."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import polars as pl

from pdmdata.datasets.xjtu_sy import inventory, load, load_many
from pdmdata.datasets.xjtu_sy import loader as xjtu_loader
from support_xjtu_sy import write_bearing


class XjtuSyTest(TestCase):
    def test_inventory_and_snapshot_rul(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_bearing(root, "Bearing2_4", 3, samples=8)
            write_bearing(root, "Bearing1_5", 2, samples=8)
            with patch.object(
                xjtu_loader, "get_settings"
            ) as settings, patch.object(
                xjtu_loader, "SAMPLES_PER_SNAPSHOT", 8
            ):
                settings.return_value.data_root = root
                summary = inventory()
                self.assertEqual(summary.height, 2)
                self.assertEqual(
                    summary.filter(pl.col("bearing") == "Bearing2_4")[
                        "snapshots"
                    ][0],
                    3,
                )
                history = load("Bearing2_4", with_rul=True)
                self.assertEqual(history["RUL"].to_list(), [2, 1, 0])
                self.assertTrue(
                    history["rms_horizontal"][-1]
                    > history["rms_horizontal"][0]
                )
                one = load("Bearing2_4", cycle=2, with_rul=True)
                self.assertEqual(one.height, 1)
                self.assertEqual(one["RUL"][0], 1)
                wave = load(
                    "Bearing2_4",
                    cycle=1,
                    with_waveform=True,
                    with_rul=True,
                )
                self.assertEqual(wave.height, 8)
                self.assertEqual(wave["RUL"].unique().to_list(), [2])
                many = load_many(
                    ["Bearing2_4", "Bearing1_5"], with_rul=True
                )
                self.assertEqual(many.height, 5)

    def test_missing_data_raises(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(xjtu_loader, "get_settings") as settings:
                settings.return_value.data_root = root
                with self.assertRaises(FileNotFoundError):
                    inventory()
                with self.assertRaises(ValueError):
                    load("Bearing9_9")
